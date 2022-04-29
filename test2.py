class A:
    def __init__(self):
        self.a = 1
        self.b = 2
        self.c = 3

    @property
    def d(self):
        return 4

    def cool(self):
        print('cool')

a = A()
print(vars(a))
