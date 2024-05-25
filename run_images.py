import subprocess


def main():
    with open("models/container_names.txt", "r") as f:
        container_names = f.read().split("\n")
        container_names = [nm for nm in container_names if nm != "" and not nm.startswith("#")] # remove empty lines and comments

    for container_name in container_names:
        print(f"Executing main.py in {container_name}")
        print(subprocess.check_output(["docker", "exec", container_name, "pip3", "install", "psutil"]).decode())
        print(
            subprocess.check_output(
                ["docker", "exec", "-e", f"MODEL_NM={container_name}", "-it", container_name, "python", "/app/main.py"]
            ).decode()
        )


if __name__ == "__main__":
    main()
