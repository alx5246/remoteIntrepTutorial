# A. Lons
# Jan 2017
#
# i AM TRYING TO figure out how to us exceptions to break out of stuff. Inside the CIFAR-10 eval, they use an Exception
# to request the coordinator to stop. I am thinking this brings all things under the coordinator to a halt. So I am
# going to see if this makes sense.... RESULT: the inside loop even if it throws an exception will not cause a break

def loop_inside(input_num):
    """
    I am going to call this inner loop which I want to have an exception which will hopefully break the outside
    :return:
    """
    try:
        some_val = 1/input_num
        print("  Inside loop val =", some_val)
    except Exception as e:
        print(type(e))
        return

def loop_outside():
    counter = 6
    while True:
        counter = counter - 1
        print("Outside loop counter =", counter)
        loop_inside(counter)
        if counter < -2:
            break

if __name__ == '__main__':
    loop_outside()