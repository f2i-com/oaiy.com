//! A WGSL compute kernel as CUDA C++.
//!
//! For a WebGPU runtime on an NVIDIA card that no WebGPU driver reaches (a Mac's eGPU, driven by tinygrad through its
//! TinyGPU app): the kernel's WGSL is parsed and validated by naga, as wgpu does, and its module walked into one
//! `extern "C" __global__` kernel, `oaiy_main`, beside a prelude of WGSL's types and built-ins (`prelude.cuh`), for
//! nvcc to compile. [`Kernel`] says what the runtime needs besides: the workgroup's size and the bindings, in the order
//! the kernel takes them.
//!
//! How WGSL's meaning is kept:
//! - every expression naga emits is a `const auto` named after it, at the place naga emits it, so its value is the
//!   one at that point (a load before a later store); a pointer is written where it is used, as an lvalue;
//! - a vector is `vec<T, N>`, laid out as WGSL lays it out; an array of fixed size a struct around its elements (so
//!   it is copied as a value); a struct its members at naga's offsets, padded between;
//! - a buffer is a pointer the kernel takes, and every function that uses one (naga's `global_uses`) takes it too; a
//!   private variable is the kernel's own, passed the same way; a workgroup variable is `__shared__`, zeroed as the
//!   kernel starts (WebGPU's default);
//! - a loop's continuing block runs as the next pass starts (naga's Metal backend's shape: `continue` is C's).
//!
//! What it does not take, it says (an `Err` naming it): images, samplers, matrices, cooperative matrices, subgroup
//! operations, overrides, `arrayLength`.

use std::collections::HashSet;
use std::fmt::Write as _;

use naga::{
    AddressSpace, ArraySize, BinaryOperator, Expression as E, Handle, Literal, MathFunction as M, Module, Scalar, ScalarKind,
    Statement as S, Type, TypeInner as T,
};

/// The prelude every kernel is compiled with.
pub const PRELUDE: &str = include_str!("prelude.cuh");

/// The name of the kernel's function.
pub const ENTRY: &str = "oaiy_main";

/// A binding's kind, as the runtime passes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingKind {
    /// `var<storage, read>`
    StorageRead,
    /// `var<storage, read_write>`
    Storage,
    /// `var<uniform>`
    Uniform,
}

/// One of the kernel's bindings: its group and number, and what it is. The kernel takes one pointer a binding, in
/// [`Kernel::bindings`]' order: the buffer's address plus the binding's offset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binding {
    pub group: u32,
    pub binding: u32,
    pub kind: BindingKind,
}

/// A WGSL compute entry point as a CUDA kernel.
#[derive(Clone, Debug)]
pub struct Kernel {
    /// The whole translation unit: the prelude and the kernel (`extern "C" __global__ void oaiy_main(...)`).
    pub source: String,
    /// The entry point's workgroup size (CUDA's block).
    pub workgroup_size: [u32; 3],
    /// The bindings, in the order the kernel takes them (by group, then number).
    pub bindings: Vec<Binding>,
}

/// `wgsl`'s compute entry point (its first, or the one called `entry`) as a CUDA kernel.
pub fn translate(wgsl: &str, entry: Option<&str>) -> Result<Kernel, String> {
    let module = naga::front::wgsl::parse_str(wgsl).map_err(|e| format!("WGSL: {}", e.emit_to_string(wgsl)))?;
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .map_err(|e| format!("validation: {}", e.emit_to_string(wgsl)))?;
    let index = match entry {
        Some(name) => module.entry_points.iter().position(|e| e.name == name).ok_or_else(|| format!("no entry point called {name}"))?,
        None => module.entry_points.iter().position(|e| e.stage == naga::ShaderStage::Compute).ok_or("no compute entry point")?,
    };
    if !module.overrides.is_empty() {
        return Err("overrides are not taken".into());
    }
    Writer { module: &module, info: &info, out: String::new() }.write(index)
}

struct Writer<'a> {
    module: &'a Module,
    info: &'a naga::valid::ModuleInfo,
    out: String,
}

/// What a function body is written within.
struct Ctx<'a> {
    function: &'a naga::Function,
    info: &'a naga::valid::FunctionInfo,
    /// The expressions given a name where they were emitted.
    named: HashSet<Handle<naga::Expression>>,
    loops: usize,
}

fn err<X>(what: impl Into<String>) -> Result<X, String> {
    Err(what.into())
}

impl<'a> Writer<'a> {
    fn write(mut self, index: usize) -> Result<Kernel, String> {
        let ep = &self.module.entry_points[index];
        if ep.stage != naga::ShaderStage::Compute {
            return err("the entry point is not a compute one");
        }
        let [wx, wy, wz] = ep.workgroup_size;
        writeln!(self.out, "#define WGSL_WG_X {wx}u\n#define WGSL_WG_Y {wy}u\n#define WGSL_WG_Z {wz}u").unwrap();
        self.out.push_str(PRELUDE);
        self.out.push_str("\n// ---- the kernel's types ----\n");
        self.write_types()?;
        // the bindings, by group and number
        let mut bound: Vec<(u32, u32, Handle<naga::GlobalVariable>)> = Vec::new();
        for (h, g) in self.module.global_variables.iter() {
            match g.space {
                AddressSpace::Storage { .. } | AddressSpace::Uniform => {
                    let b = g.binding.as_ref().ok_or("a buffer without a binding")?;
                    bound.push((b.group, b.binding, h));
                }
                AddressSpace::WorkGroup | AddressSpace::Private => {}
                other => return err(format!("a global in {other:?} is not taken")),
            }
        }
        bound.sort();
        self.out.push_str("\n// ---- the workgroup's variables ----\n");
        for (h, g) in self.module.global_variables.iter() {
            if g.space == AddressSpace::WorkGroup {
                let ty = self.type_name(g.ty)?;
                writeln!(self.out, "__shared__ __align__(16) {ty} {};", gname(h)).unwrap();
            }
        }
        self.out.push_str("\n// ---- the functions ----\n");
        for (h, f) in self.module.functions.iter() {
            let sig = self.signature(h, f)?;
            writeln!(self.out, "{sig};").unwrap();
        }
        for (h, f) in self.module.functions.iter() {
            let sig = self.signature(h, f)?;
            writeln!(self.out, "\n{sig} {{").unwrap();
            let mut ctx = Ctx { function: f, info: &self.info[h], named: HashSet::new(), loops: 0 };
            self.write_locals(&mut ctx, 1)?;
            self.write_block(&mut ctx, &f.body, 1)?;
            self.out.push_str("}\n");
        }
        // the kernel
        let f = &ep.function;
        let finfo = self.info.get_entry_point(index);
        let [x, y, z] = ep.workgroup_size;
        let mut params = Vec::new();
        let mut bindings = Vec::new();
        for (group, binding, h) in &bound {
            let g = &self.module.global_variables[*h];
            let kind = match g.space {
                AddressSpace::Uniform => BindingKind::Uniform,
                AddressSpace::Storage { access } if access.contains(naga::StorageAccess::STORE) => BindingKind::Storage,
                _ => BindingKind::StorageRead,
            };
            params.push(format!("{} {}", self.global_pointer_type(*h)?, gname(*h)));
            bindings.push(Binding { group: *group, binding: *binding, kind });
        }
        writeln!(self.out, "\nextern \"C\" __global__ void __launch_bounds__({}) {ENTRY}({}) {{", x * y * z, params.join(", ")).unwrap();
        // workgroup memory zeroed, then a barrier
        let mut zeroed = false;
        for (h, g) in self.module.global_variables.iter() {
            if g.space == AddressSpace::WorkGroup {
                writeln!(self.out, "    wgsl_zero_workgroup(&{0}, sizeof({0}));", gname(h)).unwrap();
                zeroed = true;
            }
        }
        if zeroed {
            self.out.push_str("    __syncthreads();\n");
        }
        // private variables: the kernel's own, passed on by pointer
        for (h, g) in self.module.global_variables.iter() {
            if g.space == AddressSpace::Private {
                let ty = self.type_name(g.ty)?;
                let init = match g.init {
                    Some(e) => self.const_expr(e)?,
                    None => format!("{ty}{{}}"),
                };
                writeln!(self.out, "    {ty} {0}_v = {init};\n    {ty}* {0} = &{0}_v;", gname(h)).unwrap();
            }
        }
        // the built-in arguments
        for (i, arg) in f.arguments.iter().enumerate() {
            let value = self.builtin_argument(arg)?;
            let ty = self.type_name(arg.ty)?;
            writeln!(self.out, "    const {ty} a{i} = {value};").unwrap();
        }
        let mut ctx = Ctx { function: f, info: finfo, named: HashSet::new(), loops: 0 };
        self.write_locals(&mut ctx, 1)?;
        self.write_block(&mut ctx, &f.body, 1)?;
        self.out.push_str("}\n");
        Ok(Kernel { source: self.out, workgroup_size: [x, y, z], bindings })
    }

    // ---- types ----

    fn scalar_name(s: Scalar) -> Result<&'static str, String> {
        Ok(match (s.kind, s.width) {
            (ScalarKind::Bool, _) => "bool",
            (ScalarKind::Sint, 4) | (ScalarKind::AbstractInt, _) => "int",
            (ScalarKind::Uint, 4) => "uint",
            (ScalarKind::Float, 4) | (ScalarKind::AbstractFloat, _) => "float",
            (ScalarKind::Float, 2) => "__half",
            (ScalarKind::Uint, 8) => "u64",
            (ScalarKind::Sint, 8) => "long long",
            (ScalarKind::Float, 8) => "double",
            (k, w) => return err(format!("the scalar {k:?} of {w} bytes is not taken")),
        })
    }

    fn type_name(&self, ty: Handle<Type>) -> Result<String, String> {
        Ok(match &self.module.types[ty].inner {
            T::Scalar(s) => Self::scalar_name(*s)?.to_string(),
            T::Atomic(s) => Self::scalar_name(*s)?.to_string(),
            T::Vector { size, scalar } => format!("vec<{}, {}>", Self::scalar_name(*scalar)?, *size as u8),
            T::Array { size: ArraySize::Constant(_), .. } => format!("arr_{}", ty.index()),
            T::Array { size: ArraySize::Dynamic, base, .. } => self.type_name(*base)?,
            T::Struct { .. } => format!("st_{}", ty.index()),
            other => return err(format!("the type {other:?} is not taken")),
        })
    }

    /// Every array of fixed size and struct, in naga's order (each after what it is made of).
    fn write_types(&mut self) -> Result<(), String> {
        for (h, ty) in self.module.types.iter() {
            match &ty.inner {
                T::Array { base, size: ArraySize::Constant(n), stride } => {
                    let elem = self.type_name(*base)?;
                    let elem_size = self.module.types[*base].inner.size(self.module.to_ctx());
                    if elem_size != *stride && !matches!(self.module.types[*base].inner, T::Vector { size: naga::VectorSize::Tri, .. }) {
                        return err(format!("an array of {elem} {stride} bytes apart (its elements {elem_size}) is not taken"));
                    }
                    writeln!(self.out, "struct arr_{} {{ {elem} inner[{n}]; }};", h.index()).unwrap();
                }
                T::Array { size: ArraySize::Pending(_), .. } => return err("an array sized by an override is not taken"),
                T::Struct { members, span } => {
                    let name = format!("st_{}", h.index());
                    writeln!(self.out, "struct {name} {{").unwrap();
                    let mut at = 0u32;
                    let mut params = Vec::new();
                    let mut sets = Vec::new();
                    for (i, m) in members.iter().enumerate() {
                        if m.offset > at {
                            writeln!(self.out, "    char _pad{i}[{}];", m.offset - at).unwrap();
                        }
                        let (mty, msize) = match &self.module.types[m.ty].inner {
                            // (a runtime-sized array last in a buffer's struct: its first element, the rest past it)
                            T::Array { size: ArraySize::Dynamic, base, .. } => (format!("{}", self.type_name(*base)?), 0),
                            inner => (self.type_name(m.ty)?, inner.size(self.module.to_ctx())),
                        };
                        if msize == 0 {
                            writeln!(self.out, "    {mty} m{i}[1];").unwrap();
                        } else {
                            writeln!(self.out, "    {mty} m{i};").unwrap();
                            params.push(format!("{mty} v{i}"));
                            sets.push(format!("r.m{i} = v{i};"));
                        }
                        at = m.offset + msize;
                    }
                    if *span > at {
                        writeln!(self.out, "    char _pad_end[{}];", span - at).unwrap();
                    }
                    self.out.push_str("};\n");
                    writeln!(self.out, "__device__ __forceinline__ {name} make_{name}({}) {{ {name} r{{}}; {} return r; }}", params.join(", "), sets.join(" ")).unwrap();
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// How a buffer is held: a pointer to its elements (a runtime-sized array) or to its value.
    fn global_pointer_type(&self, h: Handle<naga::GlobalVariable>) -> Result<String, String> {
        let g = &self.module.global_variables[h];
        let ty = self.type_name(g.ty)?;
        let constant = match g.space {
            AddressSpace::Uniform => true,
            AddressSpace::Storage { access } => !access.contains(naga::StorageAccess::STORE),
            _ => false,
        };
        Ok(format!("{}{ty}* __restrict__", if constant { "const " } else { "" }))
    }

    fn signature(&self, h: Handle<naga::Function>, f: &naga::Function) -> Result<String, String> {
        let ret = match &f.result {
            Some(r) => self.type_name(r.ty)?,
            None => "void".into(),
        };
        let mut params = Vec::new();
        for (i, a) in f.arguments.iter().enumerate() {
            params.push(match &self.module.types[a.ty].inner {
                T::Pointer { base, .. } => format!("{}& a{i}", self.type_name(*base)?),
                _ => format!("{} a{i}", self.type_name(a.ty)?),
            });
        }
        for g in self.passed_globals(h) {
            params.push(self.passed_param(g)?);
        }
        Ok(format!("__device__ {ret} f{}({})", h.index(), params.join(", ")))
    }

    /// The globals a function uses that the kernel holds as pointers (buffers, private variables): passed to it.
    fn passed_globals(&self, h: Handle<naga::Function>) -> Vec<Handle<naga::GlobalVariable>> {
        let fi = &self.info[h];
        self.module
            .global_variables
            .iter()
            .filter(|(g, v)| !matches!(v.space, AddressSpace::WorkGroup) && !fi[*g].is_empty())
            .map(|(g, _)| g)
            .collect()
    }

    fn passed_param(&self, g: Handle<naga::GlobalVariable>) -> Result<String, String> {
        let v = &self.module.global_variables[g];
        Ok(match v.space {
            AddressSpace::Private => format!("{}* {}", self.type_name(v.ty)?, gname(g)),
            _ => format!("{} {}", self.global_pointer_type(g)?, gname(g)),
        })
    }

    fn builtin_argument(&self, arg: &naga::FunctionArgument) -> Result<String, String> {
        match &arg.binding {
            Some(naga::Binding::BuiltIn(b)) => Self::builtin(*b),
            _ => match &self.module.types[arg.ty].inner {
                T::Struct { members, .. } => {
                    let mut values = Vec::new();
                    for m in members {
                        match &m.binding {
                            Some(naga::Binding::BuiltIn(b)) => values.push(Self::builtin(*b)?),
                            _ => return err("an entry point's argument that is not a built-in is not taken"),
                        }
                    }
                    Ok(format!("make_st_{}({})", arg.ty.index(), values.join(", ")))
                }
                _ => err("an entry point's argument that is not a built-in is not taken"),
            },
        }
    }

    fn builtin(b: naga::BuiltIn) -> Result<String, String> {
        use naga::BuiltIn as B;
        Ok(match b {
            B::GlobalInvocationId => "vec<uint, 3>(blockIdx.x * WGSL_WG_X + threadIdx.x, blockIdx.y * WGSL_WG_Y + threadIdx.y, blockIdx.z * WGSL_WG_Z + threadIdx.z)".into(),
            B::LocalInvocationId => "vec<uint, 3>(threadIdx.x, threadIdx.y, threadIdx.z)".into(),
            B::LocalInvocationIndex => "wgsl_local_index()".into(),
            B::WorkGroupId => "vec<uint, 3>(blockIdx.x, blockIdx.y, blockIdx.z)".into(),
            // (CUDA's gridDim: the driver's words in constant bank 0, which a runtime fills: tinygpu's does)
            B::NumWorkGroups => "vec<uint, 3>(gridDim.x, gridDim.y, gridDim.z)".into(),
            B::WorkGroupSize => "vec<uint, 3>(WGSL_WG_X, WGSL_WG_Y, WGSL_WG_Z)".into(),
            B::SubgroupSize => "32u".into(),
            B::SubgroupInvocationId => "(wgsl_local_index() & 31u)".into(),
            B::SubgroupId => "(wgsl_local_index() >> 5)".into(),
            B::NumSubgroups => "((WGSL_WG_X * WGSL_WG_Y * WGSL_WG_Z + 31u) >> 5)".into(),
            other => return err(format!("the built-in {other:?} is not taken")),
        })
    }

    // ---- bodies ----

    fn write_locals(&mut self, ctx: &mut Ctx, level: usize) -> Result<(), String> {
        for (h, l) in ctx.function.local_variables.iter() {
            let ty = self.type_name(l.ty)?;
            let init = match l.init {
                Some(e) => self.expr(ctx, e)?,
                None => format!("{ty}{{}}"),
            };
            writeln!(self.out, "{}{ty} l{} = {init};", indent(level), h.index()).unwrap();
        }
        Ok(())
    }

    fn write_block(&mut self, ctx: &mut Ctx, block: &naga::Block, level: usize) -> Result<(), String> {
        for st in block.iter() {
            self.write_statement(ctx, st, level)?;
        }
        Ok(())
    }

    fn write_statement(&mut self, ctx: &mut Ctx, st: &naga::Statement, level: usize) -> Result<(), String> {
        let pad = indent(level);
        match st {
            S::Emit(range) => {
                for h in range.clone() {
                    if self.is_pointer(ctx, h) {
                        continue;
                    }
                    let value = self.expr_inline(ctx, h)?;
                    writeln!(self.out, "{pad}const auto e{} = {value};", h.index()).unwrap();
                    ctx.named.insert(h);
                }
            }
            S::Block(b) => {
                writeln!(self.out, "{pad}{{").unwrap();
                self.write_block(ctx, b, level + 1)?;
                writeln!(self.out, "{pad}}}").unwrap();
            }
            S::If { condition, accept, reject } => {
                let c = self.expr(ctx, *condition)?;
                writeln!(self.out, "{pad}if ({c}) {{").unwrap();
                self.write_block(ctx, accept, level + 1)?;
                if !reject.is_empty() {
                    writeln!(self.out, "{pad}}} else {{").unwrap();
                    self.write_block(ctx, reject, level + 1)?;
                }
                writeln!(self.out, "{pad}}}").unwrap();
            }
            S::Switch { selector, cases } => {
                let s = self.expr(ctx, *selector)?;
                writeln!(self.out, "{pad}switch ({s}) {{").unwrap();
                for case in cases {
                    match case.value {
                        naga::SwitchValue::I32(v) => writeln!(self.out, "{pad}case {v}: {{").unwrap(),
                        naga::SwitchValue::U32(v) => writeln!(self.out, "{pad}case {v}u: {{").unwrap(),
                        naga::SwitchValue::Default => writeln!(self.out, "{pad}default: {{").unwrap(),
                    }
                    self.write_block(ctx, &case.body, level + 1)?;
                    if !case.fall_through && !ends_in_jump(&case.body) {
                        writeln!(self.out, "{}break;", indent(level + 1)).unwrap();
                    }
                    writeln!(self.out, "{pad}}}").unwrap();
                }
                writeln!(self.out, "{pad}}}").unwrap();
            }
            S::Loop { body, continuing, break_if } => {
                let n = ctx.loops;
                ctx.loops += 1;
                if continuing.is_empty() && break_if.is_none() {
                    writeln!(self.out, "{pad}while (true) {{").unwrap();
                } else {
                    writeln!(self.out, "{pad}bool loop_init{n} = true;\n{pad}while (true) {{\n{pad}    if (!loop_init{n}) {{").unwrap();
                    self.write_block(ctx, continuing, level + 2)?;
                    if let Some(b) = break_if {
                        let c = self.expr(ctx, *b)?;
                        writeln!(self.out, "{}if ({c}) break;", indent(level + 2)).unwrap();
                    }
                    writeln!(self.out, "{pad}    }}\n{pad}    loop_init{n} = false;").unwrap();
                }
                self.write_block(ctx, body, level + 1)?;
                writeln!(self.out, "{pad}}}").unwrap();
            }
            S::Break => writeln!(self.out, "{pad}break;").unwrap(),
            S::Continue => writeln!(self.out, "{pad}continue;").unwrap(),
            S::Return { value } => match value {
                Some(v) => {
                    let v = self.expr(ctx, *v)?;
                    writeln!(self.out, "{pad}return {v};").unwrap();
                }
                None => writeln!(self.out, "{pad}return;").unwrap(),
            },
            S::Kill => return err("discard is not a compute statement"),
            S::ControlBarrier(b) => {
                if b.contains(naga::Barrier::STORAGE) {
                    writeln!(self.out, "{pad}__threadfence();").unwrap();
                }
                if b.intersects(naga::Barrier::WORK_GROUP | naga::Barrier::STORAGE) {
                    writeln!(self.out, "{pad}__syncthreads();").unwrap();
                } else if b.contains(naga::Barrier::SUB_GROUP) {
                    writeln!(self.out, "{pad}__syncwarp();").unwrap();
                }
            }
            S::MemoryBarrier(b) => {
                if b.contains(naga::Barrier::STORAGE) {
                    writeln!(self.out, "{pad}__threadfence();").unwrap();
                } else {
                    writeln!(self.out, "{pad}__threadfence_block();").unwrap();
                }
            }
            S::Store { pointer, value } => {
                let p = self.lvalue(ctx, *pointer)?;
                let v = self.expr(ctx, *value)?;
                if self.points_to_atomic(ctx, *pointer) {
                    writeln!(self.out, "{pad}atomicExch(&{p}, {v});").unwrap();
                } else {
                    writeln!(self.out, "{pad}{p} = {v};").unwrap();
                }
            }
            S::Call { function, arguments, result } => {
                let mut args = Vec::new();
                for a in arguments {
                    args.push(if self.is_pointer(ctx, *a) { self.lvalue(ctx, *a)? } else { self.expr(ctx, *a)? });
                }
                for g in self.passed_globals(*function) {
                    args.push(gname(g));
                }
                let call = format!("f{}({})", function.index(), args.join(", "));
                match result {
                    Some(r) => {
                        writeln!(self.out, "{pad}const auto e{} = {call};", r.index()).unwrap();
                        ctx.named.insert(*r);
                    }
                    None => writeln!(self.out, "{pad}{call};").unwrap(),
                }
            }
            S::Atomic { pointer, fun, value, result } => {
                let p = self.lvalue(ctx, *pointer)?;
                let v = self.expr(ctx, *value)?;
                use naga::AtomicFunction as A;
                let call = match fun {
                    A::Add => format!("atomicAdd(&{p}, {v})"),
                    A::Subtract => format!("atomicSub(&{p}, {v})"),
                    A::And => format!("atomicAnd(&{p}, {v})"),
                    A::InclusiveOr => format!("atomicOr(&{p}, {v})"),
                    A::ExclusiveOr => format!("atomicXor(&{p}, {v})"),
                    A::Min => format!("atomicMin(&{p}, {v})"),
                    A::Max => format!("atomicMax(&{p}, {v})"),
                    A::Exchange { compare: None } => format!("atomicExch(&{p}, {v})"),
                    A::Exchange { compare: Some(c) } => {
                        let c = self.expr(ctx, *c)?;
                        let r = result.ok_or("a compare-exchange without its result")?;
                        let E::AtomicResult { ty, .. } = ctx.function.expressions[r] else { return err("a compare-exchange's result") };
                        writeln!(self.out, "{pad}const auto old{0} = atomicCAS(&{p}, {c}, {v});\n{pad}const auto e{0} = make_st_{1}(old{0}, old{0} == {c});", r.index(), ty.index()).unwrap();
                        ctx.named.insert(r);
                        return Ok(());
                    }
                };
                match result {
                    Some(r) => {
                        writeln!(self.out, "{pad}const auto e{} = {call};", r.index()).unwrap();
                        ctx.named.insert(*r);
                    }
                    None => writeln!(self.out, "{pad}{call};").unwrap(),
                }
            }
            S::WorkGroupUniformLoad { pointer, result } => {
                let p = self.lvalue(ctx, *pointer)?;
                writeln!(self.out, "{pad}__syncthreads();\n{pad}const auto e{} = {p};\n{pad}__syncthreads();", result.index()).unwrap();
                ctx.named.insert(*result);
            }
            other => return err(format!("the statement {} is not taken", short(other))),
        }
        Ok(())
    }

    // ---- expressions ----

    fn resolve<'b>(&'b self, ctx: &'b Ctx, h: Handle<naga::Expression>) -> &'b naga::TypeInner {
        ctx.info[h].ty.inner_with(&self.module.types)
    }

    fn is_pointer(&self, ctx: &Ctx, h: Handle<naga::Expression>) -> bool {
        matches!(self.resolve(ctx, h), T::Pointer { .. } | T::ValuePointer { .. })
    }

    fn points_to_atomic(&self, ctx: &Ctx, h: Handle<naga::Expression>) -> bool {
        matches!(self.resolve(ctx, h), T::Pointer { base, .. } if matches!(self.module.types[*base].inner, T::Atomic(_)))
    }

    /// An expression's value where it is used: its name, if it has one.
    fn expr(&self, ctx: &Ctx, h: Handle<naga::Expression>) -> Result<String, String> {
        if ctx.named.contains(&h) {
            return Ok(format!("e{}", h.index()));
        }
        if self.is_pointer(ctx, h) {
            return err("a pointer used as a value");
        }
        self.expr_inline(ctx, h)
    }

    /// What a pointer expression points at, as a C++ lvalue.
    fn lvalue(&self, ctx: &Ctx, h: Handle<naga::Expression>) -> Result<String, String> {
        match &ctx.function.expressions[h] {
            E::GlobalVariable(g) => {
                let v = &self.module.global_variables[*g];
                Ok(match v.space {
                    AddressSpace::WorkGroup => gname(*g),
                    // (a runtime-sized array's pointer is its first element's: indexed, not dereferenced)
                    _ if matches!(self.module.types[v.ty].inner, T::Array { size: ArraySize::Dynamic, .. }) => gname(*g),
                    _ => format!("(*{})", gname(*g)),
                })
            }
            E::LocalVariable(l) => Ok(format!("l{}", l.index())),
            E::FunctionArgument(i) => Ok(format!("a{i}")),
            E::Access { base, index } => {
                let b = self.lvalue(ctx, *base)?;
                let i = self.expr(ctx, *index)?;
                Ok(format!("{b}{}", self.subscript(self.pointee(ctx, *base)?, &i)?))
            }
            E::AccessIndex { base, index } => {
                let b = self.lvalue(ctx, *base)?;
                Ok(format!("{b}{}", self.member(self.pointee(ctx, *base)?, *index)?))
            }
            other => err(format!("the pointer {} is not taken", short(other))),
        }
    }

    /// The type a pointer expression points at.
    fn pointee<'b>(&'b self, ctx: &'b Ctx, h: Handle<naga::Expression>) -> Result<&'b naga::TypeInner, String> {
        Ok(match self.resolve(ctx, h) {
            T::Pointer { base, .. } => &self.module.types[*base].inner,
            T::ValuePointer { .. } => return err("a pointer to a vector's component indexed again"),
            _ => return err("not a pointer"),
        })
    }

    /// `[i]` into a value of `inner`'s type.
    fn subscript(&self, inner: &naga::TypeInner, i: &str) -> Result<String, String> {
        Ok(match inner {
            T::Array { size: ArraySize::Dynamic, .. } => format!("[{i}]"),
            T::Array { .. } => format!(".inner[{i}]"),
            T::Vector { .. } => format!(".c[{i}]"),
            other => return err(format!("indexing a {other:?} is not taken")),
        })
    }

    /// `.member` (or `[n]`) of a value of `inner`'s type.
    fn member(&self, inner: &naga::TypeInner, n: u32) -> Result<String, String> {
        Ok(match inner {
            T::Struct { .. } => format!(".m{n}"),
            other => self.subscript(other, &format!("{n}"))?,
        })
    }

    fn expr_inline(&self, ctx: &Ctx, h: Handle<naga::Expression>) -> Result<String, String> {
        let ex = &ctx.function.expressions[h];
        Ok(match ex {
            E::Literal(l) => literal(l)?,
            E::Constant(c) => self.const_expr(self.module.constants[*c].init)?,
            E::ZeroValue(ty) => format!("{}{{}}", self.type_name(*ty)?),
            E::Compose { ty, components } => {
                let mut parts = Vec::new();
                for c in components {
                    parts.push(self.expr(ctx, *c)?);
                }
                self.compose(*ty, &parts, components.iter().map(|c| self.resolve(ctx, *c)).collect())?
            }
            E::Splat { size, value } => {
                let v = self.expr(ctx, *value)?;
                let T::Scalar(s) = self.resolve(ctx, *value) else { return err("a splat of a non-scalar") };
                format!("vec<{}, {}>({v})", Self::scalar_name(*s)?, *size as u8)
            }
            E::Swizzle { size, vector, pattern } => {
                let v = self.expr(ctx, *vector)?;
                let T::Vector { scalar, .. } = self.resolve(ctx, *vector) else { return err("a swizzle of a non-vector") };
                let picks: Vec<String> = pattern[..*size as usize].iter().map(|p| format!("{v}.c[{}]", *p as u8)).collect();
                format!("vec<{}, {}>({})", Self::scalar_name(*scalar)?, *size as u8, picks.join(", "))
            }
            E::FunctionArgument(i) => format!("a{i}"),
            E::Access { base, index } => {
                let b = self.expr(ctx, *base)?;
                let i = self.expr(ctx, *index)?;
                format!("{b}{}", self.subscript(self.resolve(ctx, *base), &i)?)
            }
            E::AccessIndex { base, index } => {
                let b = self.expr(ctx, *base)?;
                format!("{b}{}", self.member(self.resolve(ctx, *base), *index)?)
            }
            E::Load { pointer } => self.lvalue(ctx, *pointer)?,
            E::Unary { op, expr } => {
                let v = self.expr(ctx, *expr)?;
                match op {
                    naga::UnaryOperator::Negate => format!("(-{v})"),
                    naga::UnaryOperator::LogicalNot => format!("(!{v})"),
                    naga::UnaryOperator::BitwiseNot => format!("(~{v})"),
                }
            }
            E::Binary { op, left, right } => {
                let (l, r) = (self.expr(ctx, *left)?, self.expr(ctx, *right)?);
                use BinaryOperator as B;
                let infix = |o: &str| format!("({l} {o} {r})");
                match op {
                    B::Add => infix("+"),
                    B::Subtract => infix("-"),
                    B::Multiply => infix("*"),
                    B::Divide => infix("/"),
                    B::Modulo => format!("wgsl_rem({l}, {r})"),
                    B::Equal => infix("=="),
                    B::NotEqual => infix("!="),
                    B::Less => infix("<"),
                    B::LessEqual => infix("<="),
                    B::Greater => infix(">"),
                    B::GreaterEqual => infix(">="),
                    B::And => infix("&"),
                    B::ExclusiveOr => infix("^"),
                    B::InclusiveOr => infix("|"),
                    B::LogicalAnd => infix("&&"),
                    B::LogicalOr => infix("||"),
                    B::ShiftLeft => format!("wgsl_shl({l}, {r})"),
                    B::ShiftRight => format!("wgsl_shr({l}, {r})"),
                }
            }
            E::Select { condition, accept, reject } => {
                format!("wgsl_select({}, {}, {})", self.expr(ctx, *condition)?, self.expr(ctx, *accept)?, self.expr(ctx, *reject)?)
            }
            E::Relational { fun, argument } => {
                let a = self.expr(ctx, *argument)?;
                let f = match fun {
                    naga::RelationalFunction::All => "wgsl_all",
                    naga::RelationalFunction::Any => "wgsl_any",
                    naga::RelationalFunction::IsNan => "wgsl_is_nan",
                    naga::RelationalFunction::IsInf => "wgsl_is_inf",
                };
                format!("{f}({a})")
            }
            E::Math { fun, arg, arg1, arg2, arg3 } => {
                let mut args = vec![self.expr(ctx, *arg)?];
                for a in [arg1, arg2, arg3].into_iter().flatten() {
                    args.push(self.expr(ctx, *a)?);
                }
                format!("{}({})", math_name(*fun)?, args.join(", "))
            }
            E::As { expr, kind, convert } => {
                let v = self.expr(ctx, *expr)?;
                let from = self.resolve(ctx, *expr);
                match convert {
                    Some(width) => {
                        let to = Self::scalar_name(Scalar { kind: *kind, width: *width })?;
                        format!("wgsl_convert<{to}>({v})")
                    }
                    None => {
                        let to = match from {
                            T::Scalar(s) => Self::scalar_name(Scalar { kind: *kind, width: s.width })?.to_string(),
                            T::Vector { size, scalar } => format!("vec<{}, {}>", Self::scalar_name(Scalar { kind: *kind, width: scalar.width })?, *size as u8),
                            other => return err(format!("a bitcast of {other:?}")),
                        };
                        format!("wgsl_bitcast<{to}>({v})")
                    }
                }
            }
            E::CallResult(_) | E::AtomicResult { .. } | E::WorkGroupUniformLoadResult { .. } => return err("a result used before its statement"),
            E::ArrayLength(_) => return err("arrayLength is not taken"),
            other => return err(format!("the expression {} is not taken", short(other))),
        })
    }

    /// A value made of `parts`: a vector (a vector part's components each), an array, a struct.
    fn compose(&self, ty: Handle<Type>, parts: &[String], kinds: Vec<&naga::TypeInner>) -> Result<String, String> {
        Ok(match &self.module.types[ty].inner {
            T::Vector { size, scalar } => {
                let mut flat = Vec::new();
                for (p, k) in parts.iter().zip(kinds) {
                    match k {
                        T::Vector { size: n, .. } => flat.extend((0..*n as u8).map(|i| format!("{p}.c[{i}]"))),
                        _ => flat.push(p.clone()),
                    }
                }
                format!("vec<{}, {}>({})", Self::scalar_name(*scalar)?, *size as u8, flat.join(", "))
            }
            T::Array { .. } => format!("{}{{{{{}}}}}", self.type_name(ty)?, parts.join(", ")),
            T::Struct { .. } => format!("make_st_{}({})", ty.index(), parts.join(", ")),
            other => return err(format!("composing a {other:?}")),
        })
    }

    /// A constant expression (a constant's initializer, a private variable's), from the module's own arena.
    fn const_expr(&self, h: Handle<naga::Expression>) -> Result<String, String> {
        let ex = &self.module.global_expressions[h];
        Ok(match ex {
            E::Literal(l) => literal(l)?,
            E::Constant(c) => self.const_expr(self.module.constants[*c].init)?,
            E::ZeroValue(ty) => format!("{}{{}}", self.type_name(*ty)?),
            E::Compose { ty, components } => {
                let mut parts = Vec::new();
                let mut kinds = Vec::new();
                for c in components {
                    parts.push(self.const_expr(*c)?);
                    kinds.push(self.const_kind(*c));
                }
                let kinds: Vec<&naga::TypeInner> = kinds.iter().collect();
                self.compose(*ty, &parts, kinds)?
            }
            E::Splat { size, value } => {
                let v = self.const_expr(*value)?;
                let T::Scalar(s) = self.const_kind(*value) else { return err("a constant splat of a non-scalar") };
                format!("vec<{}, {}>({v})", Self::scalar_name(s)?, *size as u8)
            }
            other => return err(format!("the constant expression {} is not taken", short(other))),
        })
    }

    /// A constant expression's type (enough of it for [`Self::compose`]).
    fn const_kind(&self, h: Handle<naga::Expression>) -> naga::TypeInner {
        match &self.module.global_expressions[h] {
            E::Literal(l) => T::Scalar(l.scalar()),
            E::Constant(c) => self.module.types[self.module.constants[*c].ty].inner.clone(),
            E::ZeroValue(ty) | E::Compose { ty, .. } => self.module.types[*ty].inner.clone(),
            E::Splat { size, value } => match self.const_kind(*value) {
                T::Scalar(s) => T::Vector { size: *size, scalar: s },
                other => other,
            },
            _ => T::Scalar(Scalar::U32),
        }
    }
}

fn gname(h: Handle<naga::GlobalVariable>) -> String {
    format!("g{}", h.index())
}

fn indent(level: usize) -> String {
    "    ".repeat(level)
}

fn short<D: std::fmt::Debug>(d: &D) -> String {
    let s = format!("{d:?}");
    s.split(|c: char| c == ' ' || c == '(' || c == '{').next().unwrap_or("?").to_string()
}

fn ends_in_jump(block: &naga::Block) -> bool {
    matches!(block.last(), Some(S::Break | S::Continue | S::Return { .. }))
}

fn literal(l: &Literal) -> Result<String, String> {
    Ok(match *l {
        Literal::F32(v) => f32_literal(v),
        Literal::F16(v) => format!("__float2half_rn({})", f32_literal(v.to_f32())),
        Literal::F64(v) => format!("{v:?}"),
        Literal::U32(v) => format!("{v}u"),
        Literal::I32(v) => if v == i32::MIN { "(-2147483647 - 1)".into() } else { format!("{v}") },
        Literal::U16(v) => format!("((unsigned short){v})"),
        Literal::I16(v) => format!("((short){v})"),
        Literal::U64(v) => format!("{v}ull"),
        Literal::I64(v) => format!("{v}ll"),
        Literal::Bool(b) => format!("{b}"),
        Literal::AbstractInt(v) => format!("{v}"),
        Literal::AbstractFloat(v) => format!("{}", f32_literal(v as f32)),
    })
}

fn f32_literal(v: f32) -> String {
    if v.is_finite() {
        let s = format!("{v:?}");
        let s = if s.contains('.') || s.contains('e') || s.contains("inf") || s.contains("NaN") { s } else { format!("{s}.0") };
        format!("{s}f")
    } else {
        format!("__int_as_float({:#x})", v.to_bits() as i32)
    }
}

fn math_name(f: naga::MathFunction) -> Result<&'static str, String> {
    Ok(match f {
        M::Abs => "wgsl_abs",
        M::Min => "wgsl_min",
        M::Max => "wgsl_max",
        M::Clamp => "wgsl_clamp",
        M::Saturate => "wgsl_saturate",
        M::Cos => "wgsl_cos",
        M::Cosh => "wgsl_cosh",
        M::Sin => "wgsl_sin",
        M::Sinh => "wgsl_sinh",
        M::Tan => "wgsl_tan",
        M::Tanh => "wgsl_tanh",
        M::Acos => "wgsl_acos",
        M::Asin => "wgsl_asin",
        M::Atan => "wgsl_atan",
        M::Atan2 => "wgsl_atan2",
        M::Ceil => "wgsl_ceil",
        M::Floor => "wgsl_floor",
        M::Round => "wgsl_round",
        M::Fract => "wgsl_fract",
        M::Trunc => "wgsl_trunc",
        M::Ldexp => "wgsl_ldexp",
        M::Exp => "wgsl_exp",
        M::Exp2 => "wgsl_exp2",
        M::Log => "wgsl_log",
        M::Log2 => "wgsl_log2",
        M::Pow => "wgsl_pow",
        M::Dot => "wgsl_dot",
        M::Dot4I8Packed => "wgsl_dot4_i8_packed",
        M::Dot4U8Packed => "wgsl_dot4_u8_packed",
        M::Sign => "wgsl_sign",
        M::Fma => "wgsl_fma",
        M::Mix => "wgsl_mix",
        M::Step => "wgsl_step",
        M::SmoothStep => "wgsl_smoothstep",
        M::Sqrt => "wgsl_sqrt",
        M::InverseSqrt => "wgsl_inverse_sqrt",
        M::CountOneBits => "wgsl_count_one_bits",
        M::ReverseBits => "wgsl_reverse_bits",
        M::ExtractBits => "wgsl_extract_bits",
        M::InsertBits => "wgsl_insert_bits",
        M::FirstTrailingBit => "wgsl_first_trailing_bit",
        M::FirstLeadingBit => "wgsl_first_leading_bit",
        M::CountLeadingZeros => "wgsl_count_leading_zeros",
        M::CountTrailingZeros => "wgsl_count_trailing_zeros",
        M::Pack2x16float => "wgsl_pack2x16float",
        M::Unpack2x16float => "wgsl_unpack2x16float",
        M::Pack4x8unorm => "wgsl_pack4x8unorm",
        M::Unpack4x8unorm => "wgsl_unpack4x8unorm",
        other => return err(format!("the built-in {other:?} is not taken")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const KERNEL: &str = "
        struct P { n: u32, scale: f32 }
        @group(0) @binding(2) var<uniform> p: P;
        @group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
        @group(1) @binding(0) var<storage, read_write> out: array<f32>;
        var<workgroup> tile: array<f32, 64>;
        var<private> seen: u32;
        fn put(i: u32, v: f32) { out[i] = v * p.scale; seen = seen + 1u; }
        @compute @workgroup_size(64, 2) fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32) {
            tile[li % 64u] = f32(li);
            workgroupBarrier();
            var s = 0.0;
            for (var k = 0u; k < 4u; k++) { s += a[id.x][k] + f32((li << (k + 30u)) >> 1u); }
            if (id.x < p.n) { put(id.x, s + tile[63u - li % 64u]); }
        }";

    #[test]
    fn a_kernel_takes_its_bindings_by_group_then_number() {
        let k = translate(KERNEL, None).unwrap();
        assert_eq!(k.workgroup_size, [64, 2, 1]);
        let order: Vec<(u32, u32, BindingKind)> = k.bindings.iter().map(|b| (b.group, b.binding, b.kind)).collect();
        assert_eq!(order, [(0, 0, BindingKind::StorageRead), (0, 2, BindingKind::Uniform), (1, 0, BindingKind::Storage)]);
        assert!(k.source.contains("extern \"C\" __global__ void __launch_bounds__(128) oaiy_main("), "{}", k.source);
    }

    #[test]
    fn the_workgroups_size_is_the_kernels_own_and_blockdim_is_not_read() {
        let k = translate(KERNEL, None).unwrap();
        assert!(k.source.starts_with("#define WGSL_WG_X 64u\n#define WGSL_WG_Y 2u\n#define WGSL_WG_Z 1u\n"));
        let body = &k.source[k.source.find("---- the kernel's types ----").unwrap()..];
        assert!(!body.contains("blockDim"), "a runtime need not fill blockDim");
        assert!(body.contains("blockIdx.x * WGSL_WG_X + threadIdx.x"));
    }

    #[test]
    fn workgroup_memory_is_shared_and_zeroed_and_a_loop_continues_as_wgsl_does() {
        let k = translate(KERNEL, None).unwrap();
        assert!(k.source.contains("__shared__ __align__(16)"));
        assert!(k.source.contains("wgsl_zero_workgroup(&"));
        // (the for loop's increment is its continuing block, run as the next pass starts)
        assert!(k.source.contains("bool loop_init0 = true;"));
        // WGSL's shifts are by the amount modulo the width
        assert!(k.source.contains("wgsl_shl(") && k.source.contains("wgsl_shr("));
    }

    #[test]
    fn a_function_is_given_the_buffers_and_private_variables_it_uses() {
        let k = translate(KERNEL, None).unwrap();
        let sig = k.source.lines().find(|l| l.starts_with("__device__ void f0(")).expect("the helper's prototype");
        assert!(sig.contains("float* __restrict__") && sig.contains("const st_") && sig.contains("uint* g"), "{sig}");
    }

    #[test]
    fn what_is_not_compute_is_refused_by_name() {
        let src = "@group(0) @binding(0) var t: texture_2d<f32>; @group(0) @binding(1) var<storage, read_write> o: array<f32>;
                   @compute @workgroup_size(1) fn main() { o[0] = textureLoad(t, vec2<i32>(0, 0), 0).x; }";
        let e = translate(src, None).unwrap_err();
        assert!(e.contains("Handle") || e.contains("Image") || e.contains("not taken"), "{e}");
        assert!(translate("@compute @workgroup_size(1) fn main() {}", Some("nope")).unwrap_err().contains("no entry point called nope"));
    }
}
