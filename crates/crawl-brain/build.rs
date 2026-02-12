fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);

    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .file_descriptor_set_path(out_dir.join("crawl_descriptor.bin"))
        .compile_protos(
            &["../../proto/crawl.proto"],
            &["../../proto"],
        )?;
    Ok(())
}
