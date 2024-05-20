import subprocess


def main():
    with open("models/container_names.txt", "r") as f:
        container_names = f.read().split("\n")[:-1]  # Remove the last empty line

    for container_name in container_names:
        print(f"Executing main.py in {container_name}")
        print(
            subprocess.check_output(
                ["docker", "exec", "-it", container_name, "python", "main.py"]
            ).decode()
        )


if __name__ == "__main__":
    main()
