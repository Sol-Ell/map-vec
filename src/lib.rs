#![feature(pointer_is_aligned_to)]
#![feature(ptr_as_ref_unchecked)]
#![feature(alloc_layout_extra)]
#![feature(box_vec_non_null)]
#![feature(allocator_api)]
#![no_std]

extern crate alloc;

use core::slice;
use core::{
    marker::PhantomData,
    mem::forget,
    ptr::{self, NonNull},
};

use alloc::alloc::{Allocator, Global};
use alloc::vec::Vec;

pub trait Map<T>: Sized {
    type Alloc: Allocator;

    fn map<U, F>(self, f: F) -> Vec<U>
    where
        F: FnMut(T) -> U,
        Self: Map<T, Alloc = Global>;
}

impl<T> Map<T> for Vec<T, Global> {
    type Alloc = Global;

    #[inline]
    fn map<U, F>(self, f: F) -> Vec<U>
    where
        F: FnMut(T) -> U,
        Self: Map<T, Alloc = Global>,
    {
        struct ClearOnPanic<'a, T, U, A: Allocator> {
            elem_ptr: NonNull<T>,
            list: &'a mut Vec<T, A>,
            _phantom: PhantomData<U>,
        }

        impl<T, U, A: Allocator> Drop for ClearOnPanic<'_, T, U, A> {
            fn drop(&mut self) {
                let len = self.list.len();
                let start = self.list.as_non_null();
                let i = unsafe { self.elem_ptr.offset_from_unsigned(start) };

                unsafe {
                    self.list.set_len(0);

                    let converted = slice::from_raw_parts_mut(start.cast::<U>().as_mut(), i);
                    if i < len {
                        let remaining = slice::from_raw_parts_mut(
                            // We gave ownership of i-th element to mapping function that panicked,
                            // therefore we add 1 to not to drop this elem. twice.
                            start.add(i).add(1).as_mut(),
                            len - i - 1,
                        );
                        ptr::drop_in_place(remaining);
                    }
                    ptr::drop_in_place(converted);
                };
            }
        }

        fn map_forward<T, U, F, A: Allocator>(list: &mut Vec<T, A>, mut f: F)
        where
            F: FnMut(T) -> U,
        {
            assert!(size_of::<T>() >= size_of::<U>());
            let mut guarded: ClearOnPanic<'_, T, U, A> = ClearOnPanic {
                elem_ptr: list.as_non_null(),
                list,
                _phantom: PhantomData::<U>,
            };

            let mut at_u = guarded.elem_ptr.cast::<U>();
            let end = unsafe { guarded.elem_ptr.add(guarded.list.len()) };
            unsafe {
                while guarded.elem_ptr < end {
                    let elem = guarded.elem_ptr.read();
                    let mapped = f(elem);

                    if size_of::<T>() == size_of::<U>() {
                        guarded.elem_ptr.cast().write(mapped);
                    } else {
                        at_u.write(mapped);
                        at_u = at_u.add(1);
                    }
                    guarded.elem_ptr = guarded.elem_ptr.add(1);
                }
            }

            forget(guarded);
        }

        fn map_backward<T, U, F, A: Allocator>(list: &mut Vec<U, A>, mut f: F)
        where
            F: FnMut(T) -> U,
        {
            assert!(size_of::<T>() < size_of::<U>());

            let start = list.as_non_null();
            let mut guarded: ClearOnPanic<'_, U, T, A> = ClearOnPanic {
                elem_ptr: unsafe { start.add(list.len()) },
                list,
                _phantom: PhantomData::<T>,
            };

            let mut elem_ptr = guarded.elem_ptr.cast::<T>();
            unsafe {
                while guarded.elem_ptr > start {
                    elem_ptr = elem_ptr.sub(1);
                    guarded.elem_ptr = guarded.elem_ptr.sub(1);

                    let elem = elem_ptr.read();
                    let mapped = f(elem);

                    guarded.elem_ptr.write(mapped)
                }
            }

            forget(guarded);
        }

        fn map_in_place<T, U, F, A: Allocator>(mut list: Vec<T, A>, f: F) -> Vec<U, A>
        where
            F: FnMut(T) -> U,
        {
            fn transmute_list<T, U, A: Allocator>(list: Vec<T, A>) -> Vec<U, A> {
                let (start, len, cap, alloc) = list.into_parts_with_alloc();
                let cap_u = convert_capacity(size_of::<T>(), size_of::<U>(), cap);

                unsafe { Vec::from_parts_in(start.cast::<U>(), len, cap_u, alloc) }
            }

            if size_of::<T>() >= size_of::<U>() {
                map_forward::<T, U, F, A>(&mut list, f);
                return transmute_list(list);
            }

            let mut converted = transmute_list(list);
            map_backward::<T, U, F, A>(&mut converted, f);

            return converted;
        }

        const fn convert_capacity(size_of_t: usize, size_of_u: usize, cap_t: usize) -> usize {
            assert!(size_of_t % size_of_u == 0 || size_of_u % size_of_t == 0);
            if size_of_t < size_of_u {
                return size_of_u / size_of_t * cap_t;
            }
            return size_of_t / size_of_u * cap_t;
        }

        let size_of_t = size_of::<T>();
        let size_of_u = size_of::<U>();
        let align_of_t = align_of::<T>();
        let align_of_u = align_of::<U>();

        if size_of_t == 0 || size_of_u == 0 {
            return self.into_iter().map(f).collect();
        }

        if size_of_t % size_of_u == 0 {
            if align_of_t >= align_of_u {
                return map_in_place(self, f);
            } else if align_of_t.abs_diff(align_of_u).ilog2() == 1 {
                if self.as_ptr().is_aligned_to(align_of_u) {
                    return map_in_place(self, f);
                }
            }
        } else if size_of_u % size_of_t == 0 {
            if align_of_t >= align_of_u {
                let cap_u = convert_capacity(size_of_t, size_of_u, self.capacity());
                if cap_u >= self.len() {
                    return map_in_place(self, f);
                }
            } else if align_of_t.abs_diff(align_of_u).ilog2() == 1 {
                if self.as_ptr().is_aligned_to(align_of_u) {
                    let cap_u = convert_capacity(size_of_t, size_of_u, self.capacity());
                    if cap_u >= self.len() {
                        return map_in_place(self, f);
                    }
                }
            }
        }

        return self.into_iter().map(f).collect();
    }
}
