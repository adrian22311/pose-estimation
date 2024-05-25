import os
import subprocess


def main():
    for folder in os.listdir("models"):
        if not os.path.isdir(os.path.join("models", folder)):
            continue
        env_files = {
            file.removeprefix(".env."): os.path.join("models", folder, file)
            for file in os.listdir(os.path.join("models", folder))
            if file.startswith(".env.")
        }
        if not env_files:
            env_files = {folder: None}
        for name, env_file in env_files.items():
            try:
                print(f"Building image {name}")
                subprocess.check_output(
                    [
                        "docker",
                        "build",
                        "-t",
                        f"{name}:latest",
                        os.path.join("models", folder),
                    ]
                )
                env = ["--env-file", env_file] if env_file else []
                print(f"Running image {name} with env file {env_file}")
                subprocess.check_output(
                    [
                        "docker",
                        "run",
                        "--name",
                        name,
                        *env,
                        "-v",
                        f"{os.path.join(os.getcwd(), 'sampled')}:/app/data:ro",
                        "-v",
                        f"{os.path.join(os.getcwd(), 'out')}:/app/out:rw",
                        "-v",
                        f"{os.path.join(os.getcwd(), 'main.py')}:/app/main.py:ro",
                        "-it",
                        "-d",
                        f"{name}:latest",
                    ]
                )
            except subprocess.CalledProcessError as e:
                print(e.output)
                continue

            with open(os.path.join("models", "container_names.txt"), "r") as f_read:
                container_names = f_read.read().split("\n")[:-1]
                if name not in container_names:
                    with open(os.path.join("models", "container_names.txt"), "a") as f_append:
                        f_append.write(f"{name}\n")


if __name__ == "__main__":
    main()
