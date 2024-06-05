import time
import logging
import psycopg2

from elements.FrameElement import FrameElement
from elements.VideoEndBreakElement import VideoEndBreakElement
from utils_local.utils import profile_time

logger = logging.getLogger(__name__)

class SendInfoDBNode:
    """Модель для отправки информации о работнике в базу данных"""
    def __init__(self, config: dict) -> None:
        config_db = config["send_info_db_node"]
        self.table_name = config_db["table_name"]
        drop_prev_table = config_db['drop_prev_table']
        self.last_db_update = time.time()
        self.memory_safe =  config_db['memory_safe'] # режим сбережения памяти
        self.time_to_clean = config_db['time_to_clean'] #за какое время от текущего должна быть очищена таблица
        self.time_to_check = config_db['time_to_check'] #как часто проверяем таблицу на необходимость очистки
        self.prev_clean_time = 0 # предыдущее время проверки необходимости очистки таблицы, в секундах
          # Параметры подключения к базе данных
        db_connection = config_db["connection_info"]
        conn_params = {
            "user": db_connection["user"],
            "password": db_connection["password"],
            "host": db_connection["host"],
            "port": str(db_connection["port"]),
            "database": db_connection["database"],
        }
        # Подключение к базе данных 
        print("conn_params", conn_params)
        try:
            self.connection = psycopg2.connect(**conn_params)
            print("Connected to PostgreSQL")
        except (Exception, psycopg2.Error) as error:
            print("Error while connecting to PostgreSQL:", error)
            raise  # Re-raise the exception after logging
        self.cursor = self.connection.cursor()

        # SQL-запрос для удаления таблицы, если она уже существует
        drop_table_query = f"DROP TABLE IF EXISTS {self.table_name};"

        if drop_prev_table:
            # Удаление таблицы, если она уже существует 
            try:
                self.cursor.execute(drop_table_query)
                self.connection.commit()
            except (Exception, psycopg2.Error) as error:
                logger.error(
                    f"Error while dropping table:: {error}"
                ) 
        # SQL-запрос для создания таблицы
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id SERIAL PRIMARY KEY,
            timestamp INTEGER,
            blinking_frequency FLOAT, 
            sleep_status BOOLEAN
        );
        """
         # Создание таблицы
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
            logger.info(
                f"Table {self.table_name} created successfully"
            ) 
        except (Exception, psycopg2.Error) as error:
            logger.error(
                f"Error while creating table: {error}"
            )
        # Запрос на удаление таблицы за заданный time_to_clean промежуток времени  
        self.clear_db = f"""
            DELETE FROM {self.table_name}
            WHERE timestamp < (
            SELECT MAX(timestamp) - INTERVAL '{self.time_to_clean} minutes'
            FROM {self.table_name});
        """
    @profile_time 
    def process(self, frame_element: FrameElement) -> FrameElement:
        # Выйти из обработки если это пришел VideoEndBreakElement а не FrameElement
        if isinstance(frame_element, VideoEndBreakElement):
            return frame_element
        assert isinstance(
            frame_element, FrameElement
        ), f"SendInfoDBNode | Неправильный формат входного элемента {type(frame_element)}"

        # Получение значений для записи в бд новой строки:
        timestamp = frame_element.timestamp

        # Преобразование sleep_status в тип bool
        sleep_status = bool(frame_element.sleep_status)

        # Определение insert_query вне зависимости от условий
        insert_query = (
            f"INSERT INTO {self.table_name} "
            "(timestamp, blinking_frequency, sleep_status) "
            "VALUES (%s, %s, %s);"
        )

        if self.memory_safe and (timestamp - self.prev_clean_time) >= (self.time_to_check * 60):
            self.prev_clean_time = timestamp
            try:
                self.cursor.execute(self.clear_db)
                self.connection.commit()
                logger.info(
                    f"Table {self.table_name} was checked for clearing successfully"
                )
            except (Exception, psycopg2.Error) as error:
                logger.error(
                    f"Error while clearing table: {error}"
                )

        try:
            self.cursor.execute(
                insert_query,
                (
                    timestamp,
                    frame_element.blinking_frequency,
                    sleep_status
                ),
            )
            self.connection.commit()
            logger.info(
                f"Successfully inserted data into PostgreSQL"
            )   
        except (Exception, psycopg2.Error) as error:
            logger.error(
                f"Error while inserting data into PostgreSQL: {error}"
            )   

        return frame_element
