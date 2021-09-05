import time


start = time.time()


while time.time() - start < 20:
    print(time.time())


print('END')