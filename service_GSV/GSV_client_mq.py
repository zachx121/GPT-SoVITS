import pika
import json
import logging
from GSV_server import rabbitmq_config,queue_service_inference_request_prefix,exchange_service_load_model_result
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 要发送的JSON数据
data = {
    "key1": "value1",
    "key2": 123,
    "key3": [1, 2, 3]
}


def connect_to_rabbitmq():
    try:
        # 创建凭证对象
        credentials = pika.PlainCredentials(rabbitmq_config["username"], rabbitmq_config["password"])
        # 创建连接参数对象
        parameters = pika.ConnectionParameters(
            host=rabbitmq_config["address"],
            port=rabbitmq_config["ports"][0],  # 默认使用第一个端口
            virtual_host=rabbitmq_config["virtual_host"],
            credentials=credentials,
            connection_attempts=3,  # 最多尝试 3 次
            retry_delay=5,         # 每次重试间隔 5 秒
            socket_timeout=10      # 套接字超时时间为 10 秒
        )
        logger.info("mq配置完毕，开始blocking connect连接")
        # 建立阻塞式连接
        connection = pika.BlockingConnection(parameters)
        logger.info("mq连接完毕，获取到connection")
        # 创建通道
        channel = connection.channel()
        logger.info("mq连接完毕，获取到channel")
        logger.info("Connected to RabbitMQ successfully.")
        return connection, channel
    except Exception as e:
        logger.error(f"Failed to connect to RabbitMQ: {repr(e)}")
        return None, None


def send_json_to_queue(connection, channel, queue_name, json_data):
    try:
        # 声明队列
        channel.queue_declare(queue=queue_name)
        # 将JSON数据转换为字符串
        message = json.dumps(json_data)
        # 设置消息属性
        properties = pika.BasicProperties(content_type='application/json')
        # 发送消息到队列
        channel.basic_publish(exchange='',
                              routing_key=queue_name,
                              body=message,
                              properties=properties)
        logger.info(f"Message sent to queue '{queue_name}': {message}")
    except Exception as e:
        logger.error(f"Failed to send message to queue: {repr(e)}")


if __name__ == "__main__":
    # 连接到RabbitMQ
    connection, channel = connect_to_rabbitmq()
    if connection and channel:
        sid = ""
        request_queue_name = queue_service_inference_request_prefix + sid

        # 队列名称，需要与服务端监听的队列名称一致
        queue_name = "your_queue_name"
        # 发送JSON数据到队列
        send_json_to_queue(connection, channel, queue_name, data)
        # 关闭连接
        connection.close()
