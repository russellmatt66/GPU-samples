import threading
import time

def print_numbers():
    for i in range(5):
        time.sleep(1)  # Simulate some work
        print(f"Number: {i}")

def print_letters():
    for letter in 'ABCDE':
        time.sleep(1)  # Simulate some work
        print(f"Letter: {letter}")

# Create threads
thread_numbers = threading.Thread(target=print_numbers)
thread_letters = threading.Thread(target=print_letters)

# Start threads
thread_numbers.start()
thread_letters.start()

# Wait for threads to finish (optional)
thread_numbers.join()
thread_letters.join()

print("Both threads have finished.")
