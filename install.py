import launch

if not launch.is_installed("transformers"):
    launch.run_pip(
        "install transformers>=4.40.0",
        "Qwen 3.5 4B Text Encoder: transformers requirement",
    )

if not launch.is_installed("safetensors"):
    launch.run_pip(
        "install safetensors",
        "Qwen 3.5 4B Text Encoder: safetensors requirement",
    )
