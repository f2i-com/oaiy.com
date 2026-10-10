//! wgsl-cuda IN.wgsl [OUT.cu]: the kernel as CUDA C++ (to OUT.cu, else stdout), and what the runtime needs of it as one
//! line of JSON on stderr: {"workgroup_size": [x, y, z], "bindings": [[group, binding, "storage"|"read"|"uniform"], ...]}.
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(input) = args.first() else {
        eprintln!("usage: wgsl-cuda IN.wgsl [OUT.cu]");
        std::process::exit(2);
    };
    let wgsl = std::fs::read_to_string(input).unwrap_or_else(|e| {
        eprintln!("{input}: {e}");
        std::process::exit(2)
    });
    let kernel = match wgsl_cuda::translate(&wgsl, None) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("{input}: {e}");
            std::process::exit(1)
        }
    };
    match args.get(1) {
        Some(out) => std::fs::write(out, &kernel.source).unwrap_or_else(|e| {
            eprintln!("{out}: {e}");
            std::process::exit(2)
        }),
        None => print!("{}", kernel.source),
    }
    let bindings: Vec<String> = kernel
        .bindings
        .iter()
        .map(|b| {
            let kind = match b.kind {
                wgsl_cuda::BindingKind::Storage => "storage",
                wgsl_cuda::BindingKind::StorageRead => "read",
                wgsl_cuda::BindingKind::Uniform => "uniform",
            };
            format!("[{}, {}, \"{kind}\"]", b.group, b.binding)
        })
        .collect();
    let [x, y, z] = kernel.workgroup_size;
    eprintln!("{{\"workgroup_size\": [{x}, {y}, {z}], \"bindings\": [{}]}}", bindings.join(", "));
}
