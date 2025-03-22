import time
import logging
from datetime import datetime, timezone, timedelta

from elements.FrameElement import FrameElement
from elements.VideoEndBreakElement import VideoEndBreakElement
from utils_local.utils import profile_time
from clickhouse_driver import Client

logger = logging.getLogger(__name__)


class SendInfoDBNode:
    """Модель для отправки информации о работнике в базу данных"""

    def __init__(self, config: dict) -> None:
        config_db = config["send_info_db_node"]
        self.table_name = config_db["table_name"]
        drop_prev_table = config_db["drop_prev_table"]
        self.last_db_update = time.time()
        self.memory_safe = config_db["memory_safe"]  # режим сбережения памяти
        self.time_to_clean = config_db[
            "time_to_clean"
        ]  # за какое время от текущего должна быть очищена таблица
        self.time_to_check = config_db[
            "time_to_check"
        ]  # как часто проверяем таблицу на необходимость очистки
        self.prev_clean_time = (
            0  # предыдущее время проверки необходимости очистки таблицы, в секундах
        )
        # Параметры подключения к базе данных
        self.db_connection = config_db["connection_info"]
        self.time_zone = config_db["time_zone"]
        # Connect to ClickHouse database
        try:
            self.client = Client(
                host=self.db_connection["host"],
                port=self.db_connection["port"],
                user=self.db_connection["user"],
                password=self.db_connection["password"],
                database=self.db_connection["database"],
            )

            logger.info("Connected to ClickHouse")
        except Exception as error:
            logger.info("Error while connecting to ClickHouse:", error)

        # SQL-запрос для удаления таблицы, если она уже существует
        drop_table_query = (
            f"DROP TABLE IF EXISTS {self.db_connection['database']}.{self.table_name};"
        )

        if drop_prev_table:
            # Удаление таблицы, если она уже существует
            try:
                self.client.execute(drop_table_query)
                logger.info(f"Table {self.table_name} dropped successfully")
            except Exception as error:
                logger.error(f"Error while dropping table: {error}")
        # SQL-запрос для создания таблицы
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.db_connection['database']}.{self.table_name} (
            timestamp DateTime, 
            blinking_frequency Nullable(Float64), 
            sleep_status Nullable(Int64),
            distance Nullable(Int64)
        ) ENGINE = MergeTree()
        ORDER BY timestamp;
        """
        # Создание таблицы
        try:
            self.client.execute(create_table_query)
            logger.info(f"Table {self.table_name} created successfully")
        except Exception as error:
            logger.error(f"Error while creating table: {error}")
        # Запрос на удаление таблицы за заданный time_to_clean промежуток времени
        self.clear_db = f"""
            ALTER TABLE {self.db_connection['database']}.{self.table_name}
            DELETE WHERE timestamp_date < now() - INTERVAL {self.time_to_clean} MINUTE;
        """

    @profile_time
    def process(self, frame_element: FrameElement) -> FrameElement:
        # Выйти из обработки если это пришел VideoEndBreakElement а не FrameElement
        if isinstance(frame_element, VideoEndBreakElement):
            return frame_element
        assert isinstance(
            frame_element, FrameElement
        ), f"SendInfoDBNode | Неправильный формат входного элемента {type(frame_element)}"

        # Преобразование sleep_status в тип bool
        sleep_status = frame_element.sleep_status
        # Определение insert_query вне зависимости от условий
        insert_query = (
            f"INSERT INTO {self.db_connection['database']}.{self.table_name} "
            "(timestamp, blinking_frequency, sleep_status, distance) "
            "VALUES"
        )

        if self.memory_safe and (frame_element.timestamp - self.prev_clean_time) >= (self.time_to_check * 60):
            self.prev_clean_time = frame_element.timestamp
            try:
                self.client.execute(self.clear_db)
                logger.info(f"Table {self.table_name} was checked for clearing successfully")
            except Exception as error:
                logger.error(f"Error while clearing table: {error}")

        # try:
        timestamp = datetime.fromtimestamp(frame_element.timestamp, tz=timezone.utc) + timedelta(
            hours=self.time_zone
        )
        data = [
            (
                timestamp,
                frame_element.blinking_frequency,
                sleep_status,
                frame_element.distance_to_the_object,
            )
        ]
        self.client.execute(insert_query, data)
        # logger.info(
        #     f"Successfully inserted data into ClickHouse"
        # )
        # except Exception as error:
        #     logger.error(
        #         f"Error while inserting data into ClickHouse: {error}"
        #     )

        return frame_element
