import json
import pika
from core.settings import settings


def publish_task(task_id: int) -> None:
    params = pika.URLParameters(settings.rabbitmq_url)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.queue_declare(queue=settings.rabbitmq_queue, durable=True)

    ch.basic_publish(
        exchange="",
        routing_key=settings.rabbitmq_queue,
        body=json.dumps({"task_id": task_id}).encode("utf-8"),
        properties=pika.BasicProperties(delivery_mode=2),
    )
    conn.close()
