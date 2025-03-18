#![feature(sized_type_properties)]
#![feature(ptr_as_ref_unchecked)]
#![feature(alloc_layout_extra)]
#![feature(box_vec_non_null)]
#![feature(allocator_api)]
#![no_std]

extern crate alloc;

use core::slice;
use core::{
    cmp::Ordering,
    marker::PhantomData,
    mem::{SizedTypeProperties, forget},
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
            /// # Panics
            ///
            /// When list is empty.
            fn drop(&mut self) {
                let len = self.list.len();
                let start = self.list.as_non_null();
                let i = unsafe { self.elem_ptr.offset_from_unsigned(start) };

                unsafe {
                    self.list.set_len(0);

                    let left = slice::from_raw_parts_mut(
                        // We gave ownership of i-th element to mapping function that panicked,
                        // therefore we add 1 to not to drop this elem. twice.
                        start.add(i).add(1).as_mut(),
                        // Panics when list is empty.
                        len - i - 1,
                    );
                    let right = slice::from_raw_parts_mut(start.cast::<U>().as_mut(), i);

                    ptr::drop_in_place(left);
                    ptr::drop_in_place(right);
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

            while guarded.elem_ptr < end {
                let elem = unsafe { guarded.elem_ptr.read() };
                let mapped = f(elem);

                unsafe {
                    if size_of::<T>() == size_of::<U>() {
                        guarded.elem_ptr.cast().write(mapped);
                    } else {
                        at_u.write(mapped);
                        at_u = at_u.add(1);
                    }
                    guarded.elem_ptr = guarded.elem_ptr.add(1);
                };
            }

            forget(guarded);
        }

        fn map_backward<T, U, F, A: Allocator>(list: &mut Vec<U, A>, mut f: F)
        where
            F: FnMut(T) -> U,
        {
            assert!(size_of::<T>() < size_of::<U>());

            let mut guarded: ClearOnPanic<'_, U, T, A> = ClearOnPanic {
                elem_ptr: list.as_non_null(),
                list,
                _phantom: PhantomData::<T>,
            };

            let mut elem_ptr = guarded.elem_ptr.cast::<T>();
            let end = unsafe { guarded.elem_ptr.add(guarded.list.len()) };

            while guarded.elem_ptr < end {
                let elem = unsafe { elem_ptr.read() };
                let mapped = f(elem);

                unsafe {
                    guarded.elem_ptr.write(mapped);

                    guarded.elem_ptr = guarded.elem_ptr.add(1);
                    elem_ptr = elem_ptr.add(1);
                }
            }

            forget(guarded);
        }

        fn next_synced_capacity_info<T, U>(cap_t: usize) -> (usize, usize) {
            const fn gcd(mut a: usize, mut b: usize) -> usize {
                while b != 0 {
                    (a, b) = (b, a % b);
                }
                return a;
            }

            let (num_t_per_lcm, num_u_per_lcm) = const {
                let size_t = size_of::<T>();
                let size_u = size_of::<U>();

                // The first synced capacity in terms of bytes is least common multiple of sizes of T and U.
                let lcm = size_t / gcd(size_t, size_u) * size_u;
                (lcm / size_t, lcm / size_u)
            };

            let number_of_elements_until_next_synced_capacity =
                num_t_per_lcm - cap_t % num_t_per_lcm;
            let cap_u = (cap_t + number_of_elements_until_next_synced_capacity) / num_u_per_lcm;

            return (number_of_elements_until_next_synced_capacity, cap_u);
        }

        fn map_in_place<T, U, F, A: Allocator>(mut list: Vec<T, A>, cap_u: usize, f: F) -> Vec<U, A>
        where
            F: FnMut(T) -> U,
        {
            fn transmute_list<T, U, A: Allocator>(list: Vec<T, A>, cap_u: usize) -> Vec<U, A> {
                let (start, len, _, alloc) = list.into_parts_with_alloc();
                unsafe { Vec::from_parts_in(start.cast::<U>(), len, cap_u, alloc) }
            }

            let size_of_t = size_of::<T>();
            let size_of_u = size_of::<U>();
            match size_of_t.cmp(&size_of_u) {
                Ordering::Less => {
                    let mut converted = transmute_list(list, cap_u);
                    map_backward::<T, U, F, A>(&mut converted, f);
                    converted
                }
                Ordering::Equal | Ordering::Greater => {
                    map_forward::<T, U, F, A>(&mut list, f);
                    transmute_list(list, cap_u)
                }
            }
        }

        let align_t = align_of::<T>();
        let align_u = align_of::<U>();
        match align_t.cmp(&align_u) {
            Ordering::Equal => {
                let len = self.len();
                let next_synced_cap = if !T::IS_ZST && !U::IS_ZST {
                    let (until_next_synced, cap_u) =
                        next_synced_capacity_info::<T, U>(self.capacity());

                    if until_next_synced == 0 && cap_u >= len {
                        return map_in_place(self, cap_u, f);
                    }

                    next_synced_capacity_info::<U, T>(len).1
                } else {
                    len
                };

                // Unfortunately, the capacity of the mapped list may be greater than requested.
                let mut mapped = Vec::with_capacity(next_synced_cap);
                mapped.extend(self.into_iter().map(f));

                return mapped;
            }
            _ => todo!(), //self.into_iter().map(f).collect(),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::Map;

    extern crate std;

    use std::{vec, vec::Vec};

    #[repr(align(2))]
    #[derive(Debug, PartialEq, Eq)]
    struct A(u8);

    #[derive(Debug, PartialEq, Eq)]
    struct B(#[allow(dead_code)] u16);

    #[test]
    fn equal_layout() {
        let list = vec![A(123_u8), A(213), A(132)];
        let mapped = list.map(|t| B(t.0 as u16));

        assert_eq!(mapped, vec![B(123), B(213), B(132)]);
    }

    #[test]
    fn greater_size() {
        let list = vec![123_u8, 213, 132];
        let mapped = list.map(A);

        assert_eq!(mapped, vec![A(123), A(213), A(132)]);
    }

    #[test]
    fn smaller_size() {
        let list = vec![B(123), B(213), B(132)];
        let mapped = list.map(|t| A(t.0 as _));

        assert_eq!(mapped, vec![A(123), A(213), A(132)]);
    }

    #[test]
    fn empty_list() {
        let list: Vec<A> = vec![];
        let mapped: Vec<B> = list.map(|_| panic!("Testing behavior at empty list."));

        assert!(mapped.is_empty());
    }

    fn zst_to_non_zst() {}

    #[test]
    fn catch_panic() {
        let list = vec![A(123_u8), A(213), A(132)];

        let mapped = std::panic::catch_unwind(|| {
            list.map(|t| {
                if t.0 == 213 {
                    panic!("Testing behavior at panic.")
                } else {
                    t
                }
            })
        });

        assert!(mapped.is_err());
    }
}
