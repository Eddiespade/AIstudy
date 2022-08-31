import threading  # 导入线程的库
import random
import time

gMoney = 0
gCondition = threading.Condition()
gTime = 0


class Producer(threading.Thread):  # 定义生产者类
    def run(self) -> None:
        global gMoney  # 定义为全球变量
        global gTime
        while True:
            gCondition.acquire()  # 获取线程
            if gTime >= 10:  # 如果生产者生产次数大于10
                gCondition.release()  # 释放线程并推出循环
                break
            money = random.randint(0, 100)  # 生产的钱在0，100之间随机取
            gMoney += money  # 赋值给生产的钱
            gTime += 1  # 生产次数加一
            print("%s生产了%d元" % (threading.current_thread().name, money))  ##生产者的名字和生产的钱
            gCondition.notify_all()  # 唤醒所有等待的线程
            gCondition.release()  # 释放掉当前线程
            time.sleep(1)


# 整个过程是获取线程-唤醒所有等待的其他线程-然后把当前线程释放掉（在取获取线程这样的一个过程）
class Consumer(threading.Thread):
    def run(self) -> None:
        global gMoney  # 消费者主要负责花钱，直至花完，所以不用定义次数
        while True:  # 进入循环
            gCondition.acquire()  # 获取次数
            money = random.randint(0, 100)  # 花钱的数量
            while gMoney < money:  # 当剩的钱不足以支撑花的钱的时候
                if gTime >= 10:  # 消费者消费大于等于10次后
                    print("%s想要消费%d元，但是目前只有%d元,生产者已经不在生产！" % (threading.current_thread().name, money, gMoney))
                    gCondition.release()
                    return
                print("%s想要消费%d元，但是目前只有%d元,消费失败！" % (
                threading.current_thread().name, money, gMoney))  # 不大于10次时候就不足以支持消费者花销，打印消费失败！
                gCondition.wait()  # 消费者线程等待，等待生产者为他生产
            gMoney -= money
            print("%s想要消费%d元，目前剩余只有%d元" % (threading.current_thread().name, money, gMoney))
            gCondition.release()
            time.sleep(1)


def main():
    for i in range(5):
        th = Producer(name='生产者%d号' % i)
        th.start()  # 制造5个生产者
    for i in range(5):
        th = Consumer(name="消费者%d号" % i)
        th.start()  # 制造5个消费者者


if __name__ == "__main__":
    main()
