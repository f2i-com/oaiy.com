//! For the tests: how much memory a piece of code holds at its most, counted per thread.
//!
//! A counting allocator that only counts on a thread that asked to be counted, so tests that run
//! side by side do not add to each other's numbers. It counts the bytes live at once (what is
//! allocated less what is freed) and keeps the highest.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static LIVE: Cell<isize> = const { Cell::new(0) };
    static PEAK: Cell<isize> = const { Cell::new(0) };
}

pub struct Counting;

fn note(delta: isize) {
    // A thread that is being torn down cannot be asked: it is not being counted.
    let _ = COUNTING.try_with(|counting| {
        if counting.get() {
            let _ = LIVE.try_with(|live| {
                let now = live.get() + delta;
                live.set(now);
                let _ = PEAK.try_with(|peak| {
                    if now > peak.get() {
                        peak.set(now);
                    }
                });
            });
        }
    });
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            note(layout.size() as isize);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            note(layout.size() as isize);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        note(-(layout.size() as isize));
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            note(new_size as isize - layout.size() as isize);
        }
        new
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Run `work` on this thread and say, with its result, the most memory it held at once (in bytes) and how
/// long it took.
pub fn measured<T>(work: impl FnOnce() -> T) -> (T, usize, std::time::Duration) {
    LIVE.with(|l| l.set(0));
    PEAK.with(|p| p.set(0));
    COUNTING.with(|c| c.set(true));
    let started = std::time::Instant::now();
    let result = work();
    let elapsed = started.elapsed();
    COUNTING.with(|c| c.set(false));
    let peak = PEAK.with(|p| p.get()).max(0) as usize;
    (result, peak, elapsed)
}

#[test]
fn what_is_held_at_once_is_counted_and_what_was_freed_is_not() {
    let ((), peak, _) = measured(|| {
        let big = vec![1u8; 8 << 20];
        assert_eq!(big.len(), 8 << 20);
        drop(big);
        let small = vec![1u8; 1 << 20];
        assert_eq!(small.len(), 1 << 20);
    });
    assert!((8 << 20..9 << 20).contains(&peak), "the 8 MiB held once is the peak, not 9 MiB added up: {peak}");
    let ((), none, _) = measured(|| {});
    assert!(none < 1 << 16, "{none}");
}
