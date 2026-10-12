use std::{env, path::PathBuf, process::Command};

fn run(command: &mut Command) {
    assert!(command.status().expect("run C compiler").success());
}

fn main() {
    if env::var_os("CARGO_FEATURE_GLIBC_VECTOR_MATH").is_none() {
        return;
    }
    println!("cargo:rerun-if-changed=src/cpu/glibc_vector_math.c");
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let object = out.join("glibc_vector_math.o");
    let archive = out.join("libcandle_glibc_vector_math.a");
    run(Command::new("cc")
        .args([
            "-O3",
            "-mavx512f",
            "-c",
            "src/cpu/glibc_vector_math.c",
            "-o",
        ])
        .arg(&object));
    run(Command::new("ar").args(["crus"]).arg(&archive).arg(&object));
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=candle_glibc_vector_math");
    println!("cargo:rustc-link-lib=mvec");
}
