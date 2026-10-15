import modal


def main():
    print("[*] Scanning Modal Volume 'acne-model-storage'...\n")
    vol = modal.Volume.from_name("acne-model-storage")

    try:
        # Retrieve all files/directories at volume root
        entries = vol.listdir("/")

        if not entries:
            print("[!] Volume is empty. No files or checkpoints found.")
            return

        for entry in entries:
            print(f"Directory: {entry.path}")

            # If directory, inspect subdirectories
            if entry.type == modal.volume.FileEntryType.DIRECTORY:
                try:
                    sub_entries = vol.listdir(entry.path)
                    for sub in sub_entries:
                        print(f"   |-- File: {sub.path}")

                        # Check one level deeper (for nested checkpoint folders)
                        if sub.type == modal.volume.FileEntryType.DIRECTORY:
                            nested_entries = vol.listdir(sub.path)
                            for nested in nested_entries:
                                print(f"       |-- Target: {nested.path}")
                except Exception:
                    pass

    except Exception as e:
        print(f"[X] Error reading Modal volume: {e}")


if __name__ == "__main__":
    main()