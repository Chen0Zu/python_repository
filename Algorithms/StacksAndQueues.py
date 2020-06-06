# Node class
class Node:
    def __init__(self, item):
        self.item = item
        self.next = None


# linked stack of strings
class linkedStackOfStrings:
    def __init__(self):
        self.head = None

    # is empty stack
    def isEmpty(self):
        return self.head == None

    # push new item into the stack
    def push(self, item):
        first = Node(item)
        first.next = self.head
        self.head = first

    def pop(self):
        first = self.head
        if first is not None:
            self.head = first.next
            return first.item
        else:
            return None

    def print(self):
        current = self.head
        while (current):
            print(current.item)
            current = current.next


class linkedQueueOfStrings:
    def __init__(self):
        self.head = None
        self.tail = None

    def isEmpty(self):
        return self.head == None

    def enqueue(self, item):
        last = Node(item)
        last.next = None
        self.tail = last
        if self.isEmpty():
            self.head = self.tail

    def dequeue(self):
        if self.isEmpty() == False:
            first = self.head
            self.head = first.next
            return first.item
        else:
            return None

    def print(self):
        current = self.head
        while current:
            print(current.item)
            current = current.next


# Arithmetic expression evaluation
expression = '(1+((2+3)*(4*5)))'
ops = linkedStackOfStrings()
vals = linkedStackOfStrings()
for ch in expression:
    if ch == '(':
        None
    elif ch == '+':
        ops.push(ch)
    elif ch == '*':
        ops.push(ch)
    elif ch == ')':
        op = ops.pop()
        if op == '+':
            vals.push(str(int(vals.pop()) + int(vals.pop())))
        elif op == '*':
            vals.push(str(int(vals.pop()) * int(vals.pop())))
    else:
        vals.push(ch)

print('The final result: ' + str(vals.pop()))
