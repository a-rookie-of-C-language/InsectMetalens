import threading
import time


def calculate_data(i):
    print("i:", i)
    print("##", i)
    time.sleep(10)


if __name__ == '__main__':
    threads = []
    for i in range(10):
        t = threading.Thread(target=calculate_data, args=(i,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
