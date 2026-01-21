from pynput import keyboard

# Define the keys we want to monitor
TARGET_KEYS = {'w', 'a', 's', 'd'}

def on_press(key):
    try:
        # Check if the alphanumeric character is in our target set
        if hasattr(key, 'char') and key.char in TARGET_KEYS:
            print(f"Key Pressed: {key.char}")
    except AttributeError:
        # This handles special keys (like Shift or Ctrl) which don't have a .char attr
        pass

def on_release(key):
    if hasattr(key, 'char') and key.char in TARGET_KEYS:
        print(f"Key Released: {key.char}")
    
    # Optional: Stop listener if 'Esc' is pressed
    if key == keyboard.Key.esc:
        print("Exiting...")
        return False

# Setup the listener
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    print("Listening for W, A, S, D... (Press ESC to stop)")
    listener.join()
