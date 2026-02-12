//! Build automation for the crawl system.
//!
//! Usage: cargo xtask <command>

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use std::path::Path;
use std::process::Command;

#[derive(Parser)]
#[command(name = "xtask", about = "Build automation for crawl")]
struct Cli {
    #[command(subcommand)]
    command: XtaskCommand,
}

#[derive(Subcommand)]
enum XtaskCommand {
    /// Build all WASM plugins targeting wasm32-wasip2.
    BuildPlugins,
    /// Build plugins and copy .wasm files to the plugins/ directory.
    InstallPlugins,
    /// Download/convert ONNX models.
    Models,
    /// Run all tests (host + plugins).
    Test,
    /// Run benchmarks.
    Bench,
    /// Show system status (toolchain, models, plugins).
    Status,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        XtaskCommand::BuildPlugins => build_plugins(),
        XtaskCommand::InstallPlugins => install_plugins(),
        XtaskCommand::Models => download_models(),
        XtaskCommand::Test => run_tests(),
        XtaskCommand::Bench => run_benchmarks(),
        XtaskCommand::Status => show_status(),
    }
}

const PLUGIN_CRATES: &[&str] = &[
    "crates/crawl-plugin-sysmon",
    "crates/crawl-plugin-logwatch",
    "crates/crawl-plugin-identifier",
    "crates/crawl-plugin-trainer",
];

fn build_plugins() -> Result<()> {
    for crate_path in PLUGIN_CRATES {
        let manifest = format!("{crate_path}/Cargo.toml");
        if !Path::new(&manifest).exists() {
            eprintln!("  skipping {crate_path} (no Cargo.toml)");
            continue;
        }

        println!("  building {crate_path} for wasm32-wasip2...");
        let status = Command::new("cargo")
            .args([
                "build",
                "--manifest-path",
                &manifest,
                "--target",
                "wasm32-wasip2",
                "--release",
            ])
            .status()
            .with_context(|| format!("failed to build {crate_path}"))?;

        if !status.success() {
            bail!("build failed for {crate_path}");
        }
    }

    println!("all plugins built successfully");
    Ok(())
}

fn install_plugins() -> Result<()> {
    build_plugins()?;

    let dest = Path::new("plugins");
    std::fs::create_dir_all(dest)?;

    for crate_path in PLUGIN_CRATES {
        let crate_name = Path::new(crate_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .replace('-', "_");

        let wasm_src = Path::new(crate_path)
            .join("target/wasm32-wasip2/release")
            .join(format!("{crate_name}.wasm"));

        if wasm_src.exists() {
            let wasm_dest = dest.join(format!("{crate_name}.wasm"));
            std::fs::copy(&wasm_src, &wasm_dest).with_context(|| {
                format!(
                    "failed to copy {} to {}",
                    wasm_src.display(),
                    wasm_dest.display()
                )
            })?;
            let size = std::fs::metadata(&wasm_dest)?.len();
            println!(
                "  installed {} ({:.1} KB)",
                wasm_dest.display(),
                size as f64 / 1024.0
            );
        } else {
            eprintln!("  warning: {} not found", wasm_src.display());
        }
    }

    println!("plugins installed to {}", dest.display());
    Ok(())
}

fn download_models() -> Result<()> {
    let models_dir = Path::new("models");
    std::fs::create_dir_all(models_dir)?;

    println!("ONNX Model Setup");
    println!("================");
    println!();

    let models = [
        (
            "tcn-ae-metrics.onnx",
            "TCN-AE for system metrics anomaly detection (~200K params)",
            "Custom trained — see docs/models.md for training script",
        ),
        (
            "gte-small.onnx",
            "GTE-small for text embeddings (33M params, ~70MB)",
            "optimum-cli export onnx --model thenlper/gte-small models/ --task feature-extraction",
        ),
        (
            "lightlog-tcn.onnx",
            "LightLog TCN for log anomaly detection (~500K params)",
            "Custom trained — see docs/models.md for training script",
        ),
        (
            "deberta-v3-small-pi.onnx",
            "DeBERTa-v3-small for prompt injection detection (44M params, ~90MB)",
            "optimum-cli export onnx --model protectai/deberta-v3-small-prompt-injection-v2 models/ --task text-classification",
        ),
    ];

    for (filename, description, instructions) in &models {
        let path = models_dir.join(filename);
        let status = if path.exists() { "FOUND" } else { "MISSING" };
        println!("  [{status}] {filename}");
        println!("    {description}");
        if !path.exists() {
            println!("    Install: {instructions}");
        }
        println!();
    }

    println!("Total estimated model footprint: ~164 MB RAM (INT8)");
    println!();
    println!("Prerequisites: pip install optimum[exporters] onnxruntime");

    Ok(())
}

fn run_tests() -> Result<()> {
    println!("running workspace tests...");
    let status = Command::new("cargo")
        .args(["test", "--workspace"])
        .status()
        .context("failed to run tests")?;

    if !status.success() {
        bail!("tests failed");
    }

    println!("all tests passed");
    Ok(())
}

fn run_benchmarks() -> Result<()> {
    println!("running benchmarks...");
    let status = Command::new("cargo")
        .args(["bench", "--workspace"])
        .status()
        .context("failed to run benchmarks")?;

    if !status.success() {
        bail!("benchmarks failed");
    }
    Ok(())
}

fn show_status() -> Result<()> {
    println!("crawl system status");
    println!("===================");

    // Toolchain.
    println!("\nRust toolchain:");
    let _ = Command::new("rustc").arg("--version").status();

    // WASM target.
    println!("\nWASM target:");
    let output = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()?;
    let targets = String::from_utf8_lossy(&output.stdout);
    if targets.contains("wasm32-wasip2") {
        println!("  wasm32-wasip2: INSTALLED");
    } else {
        println!("  wasm32-wasip2: MISSING (run: rustup target add wasm32-wasip2)");
    }

    // Models.
    println!("\nONNX models (in models/):");
    for model in &[
        "tcn-ae-metrics.onnx",
        "gte-small.onnx",
        "lightlog-tcn.onnx",
        "deberta-v3-small-pi.onnx",
    ] {
        let path = Path::new("models").join(model);
        if path.exists() {
            let size = std::fs::metadata(&path)?.len();
            println!("  {model}: {:.1} MB", size as f64 / 1024.0 / 1024.0);
        } else {
            println!("  {model}: MISSING");
        }
    }

    // Plugins.
    println!("\nBuilt plugins (in plugins/):");
    if Path::new("plugins").exists() {
        for entry in std::fs::read_dir("plugins")? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "wasm") {
                let size = std::fs::metadata(&path)?.len();
                println!(
                    "  {}: {:.1} KB",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    size as f64 / 1024.0
                );
            }
        }
    } else {
        println!("  (none — run: cargo xtask install-plugins)");
    }

    // Ollama.
    println!("\nOllama:");
    match Command::new("ollama").arg("list").output() {
        Ok(output) if output.status.success() => {
            let list = String::from_utf8_lossy(&output.stdout);
            for line in list.lines().take(5) {
                println!("  {line}");
            }
        }
        _ => println!("  NOT RUNNING (start with: ollama serve)"),
    }

    Ok(())
}
