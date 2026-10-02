//! Reviewer's measurement (F2): the memory this crate's JSON parser holds for a hostile body of a given size, with a counting allocator. The HTTP layer caps a response at
//! 1 MiB + 64 KiB (`client::relay::MAX_RESPONSE_BYTES`), so the figure that matters is the peak for about 1.1 MB of input; larger inputs show that it is linear. One test only:
//! the allocator counts the whole process. Run with `-- --nocapture --test-threads 1` to read the table.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use oaiy_relay_core::json;

struct Counting;
static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            let now = CURRENT.fetch_add(l.size(), Ordering::SeqCst) + l.size();
            PEAK.fetch_max(now, Ordering::SeqCst);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l);
        CURRENT.fetch_sub(l.size(), Ordering::SeqCst);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = System.realloc(p, l, new);
        if !q.is_null() {
            if new >= l.size() {
                let now = CURRENT.fetch_add(new - l.size(), Ordering::SeqCst) + (new - l.size());
                PEAK.fetch_max(now, Ordering::SeqCst);
            } else {
                CURRENT.fetch_sub(l.size() - new, Ordering::SeqCst);
            }
        }
        q
    }
}

#[global_allocator]
static A: Counting = Counting;

fn measure(name: &str, input: &[u8]) {
    let base = CURRENT.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    let t = Instant::now();
    let r = json::parse(input);
    let peak = PEAK.load(Ordering::SeqCst) - base;
    let kept = CURRENT.load(Ordering::SeqCst) - base;
    println!(
        "{name:<34} input {:>10} B  result {:<10}  peak extra {:>12} B ({:>6.1}x)  held {:>12} B  {:?}",
        input.len(),
        if r.is_ok() { "ok" } else { "err" },
        peak,
        peak as f64 / input.len() as f64,
        kept,
        t.elapsed()
    );
    drop(r);
}

#[test]
fn memory_for_hostile_json() {
    println!("size_of::<Json>() = {}", std::mem::size_of::<json::Json>());
    for mb in [1usize, 4] {
        let n = mb * 1_048_576;
        let mut zeros = String::from("[");
        while zeros.len() < n {
            zeros.push_str("0,");
        }
        zeros.push_str("0]");
        measure(&format!("array of 0 ({mb} MiB)"), zeros.as_bytes());
        drop(zeros);

        let mut empties = String::from("[");
        while empties.len() < n {
            empties.push_str("[],");
        }
        empties.push_str("[]]");
        measure(&format!("array of [] ({mb} MiB)"), empties.as_bytes());
        drop(empties);

        let mut obj = String::from("{");
        let mut i = 0u64;
        while obj.len() < n {
            obj.push_str(&format!("\"{i}\":0,"));
            i += 1;
        }
        obj.push_str("\"z\":0}");
        measure(&format!("object of small members ({mb} MiB)"), obj.as_bytes());
        drop(obj);

        let mut nulls = String::from("[");
        while nulls.len() < n {
            nulls.push_str("null,");
        }
        nulls.push_str("null]");
        measure(&format!("array of null ({mb} MiB)"), nulls.as_bytes());
        drop(nulls);

        let mut s = String::from("\"");
        s.push_str(&"a".repeat(n));
        s.push('"');
        measure(&format!("one string ({mb} MiB)"), s.as_bytes());
        drop(s);

        let mut big = String::new();
        big.push_str(&"9".repeat(n));
        measure(&format!("one number literal ({mb} MiB)"), big.as_bytes());
        drop(big);

        let mut esc = String::from("\"");
        while esc.len() < n {
            esc.push_str("\\u0041");
        }
        esc.push('"');
        measure(&format!("string of escapes ({mb} MiB)"), esc.as_bytes());
        drop(esc);
    }
}

