import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from aiogram.filters import CommandStart, Command
import httpx
from openai import AsyncOpenAI  # Для асинхронной работы с OpenAI
from aiogram import Bot, Dispatcher, F, types


class OpenAIClient:
    """
    Класс-обёртка над асинхронным клиентом OpenAI.

    Используем AsyncOpenAI(api_key=..., http_client=httpx.AsyncClient(), base_url=...).
    Параметры:
     - api_key: ключ OpenAI
     - base_url: адрес API (по умолчанию 'https://api.openai.com/v1')
     - model_name: строка c названием модели, например 'gpt-3.5-turbo', 'gpt-4' и т.д.

    Ожидаем, что в этом окружении установлена библиотека:
       pip install openai httpx
       (и доступен from openai import AsyncOpenAI)

    В методе analyze_message используем асинхронный метод:
       await self.client.chat.completions.create(...)

    """
    def __init__(self, api_key: str, base_url: str, model_name: str = "gpt-3.5-turbo"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_name = model_name
        self.client = AsyncOpenAI(
            api_key=api_key,
            http_client=httpx.AsyncClient(),
            base_url=base_url
        )

    async def analyze_message(self, text: str, same_topic_list: str) -> Optional[dict]:
        """
        Асинхронно отправляем запрос в OpenAI через self.client.

        Возвращает JSON-объект вида:
        {
          "has_deadline": bool,
          "deadline_title": str,
          "deadline_datetime": str (YYYY-MM-DD или YYYY-MM-DD HH:MM)
        }
        или None, если не удалось распарсить.
        """
        cur_date = datetime.now().strftime("%Y-%m-%d")
        system_prompt = (
            f"Текущая дата {cur_date} "
            "You are a helpful assistant that extracts deadline information from text messages. "
            "Return EXACTLY a JSON object with the fields: 'has_deadline' (bool), 'deadline_title' (str), "
            "'deadline_datetime' (str). 'deadline_datetime' (datetime, format YYYY-MM-DD) may be empty if no date is recognized."
            "Если в сообщении встречается мягкий и жёсткий дедлайн, записывай мягкий"
            
        )
        # user_prompt = (
        #     f"User message:\n{text}\n\n"
        # )
        # Если есть список дедлайнов из той же темы, добавим:
        if same_topic_list:
            system_prompt += ("Below is the current list of deadlines in this same topic:"\
                            f"\n{same_topic_list}\n")
        try:
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                model=self.model_name,
                temperature=0.0,
                max_tokens=500,
            )
            raw_answer = response.choices[0].message.content.strip()
            self.logger.info(f"LLM raw answer: {raw_answer}")

            try:
                parsed = json.loads(raw_answer)
                if all(k in parsed for k in ("has_deadline", "deadline_title", "deadline_datetime")):
                    return parsed
                else:
                    self.logger.warning("JSON не содержит необходимых ключей.")
                    return None
            except json.JSONDecodeError:
                self.logger.warning("Не удалось декодировать JSON из ответа LLM.")
                return None
        except Exception as e:
            self.logger.error(f"Ошибка при запросе к OpenAI: {e}")
            return None


class TelegramDeadlineBot:
    """
    Класс, реализующий Telegram-бота для чтения сообщений в (супер)группах
    и определения дедлайнов с помощью LLM (OpenAI). Бот не пишет в группы,
    только читает и анализирует.

    В конфиге можно настроить:
     - time_offset: смещение времени (timedelta), чтобы бот считал текущее время
       не от системных часов, а с заданным сдвигом.
     - chats_and_topics: словарь вида { chat_id: [topic_id1, topic_id2, ...] },
       где chat_id – ID супергруппы, а список – топики, которые нужно отслеживать.
       Если список пуст, считаем, что нужно отслеживать все сообщения в чате.

    Все данные о дедлайнах хранятся в памяти в self.deadlines_storage.
    Предполагается, что при перезапуске бота данные теряются.

    Также реализована группировка дедлайнов по чату и топику, а при выводе
    отображаются "название чата" и "название топика" (если доступно).
    """

    def __init__(
        self,
        telegram_token: str,
        openai_client: OpenAIClient,
        config: Optional[Dict[str, Any]] = None
    ):
        # -----------------------
        # Логгирование
        # -----------------------
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        # -----------------------
        # Инициализация бота и OpenAI
        # -----------------------
        self.telegram_token = telegram_token
        self.openai_client = openai_client  # наш асинхронный клиент OpenAI

        # -----------------------
        # Чтение конфига
        # -----------------------
        if config is None:
            config = {}
        self.config = config
        # Сдвиг времени (timedelta) или None
        self.time_offset: Optional[timedelta] = config.get("time_offset", None)
        # Словарь { chat_id: [topic_id1, topic_id2, ...] }
        # Если список для chat_id пуст, значит отслеживаем все сообщения в этом чате.
        self.chats_and_topics: Dict[int, List[int]] = config.get("chats_and_topics", {})

        # -----------------------
        # Инициализация бота и диспетчера
        # -----------------------
        self.bot = Bot(token=self.telegram_token)
        self.dp = Dispatcher()

        # -----------------------
        # Хранилища в памяти
        # -----------------------
        # Пример структуры deadlines_storage:
        # {
        #   "deadline_id_1": {
        #       "chat_id": int,
        #       "chat_title": str,
        #       "topic_id": Optional[int],
        #       "topic_name": Optional[str],
        #       "message_id": int,
        #       "message_link": str,
        #       "title": str,
        #       "deadline_dt": datetime,
        #       "history": [ (datetime, "Создан/Обновлён", ссылка), ... ]
        #   },
        #   ...
        # }
        self.deadlines_storage: Dict[str, Dict] = {}

        # Флаг "бот работает" (по желанию)
        self.bot_running = True

        # Регистрация хендлеров
        self.register_handlers()

    def get_current_time(self) -> datetime:
        """
        Возвращает текущее время с учетом пользовательского смещения (если задано).
        """
        now = datetime.now()
        if self.time_offset:
            return now + self.time_offset
        return now

    def register_handlers(self):
        """
        Регистрируем хендлеры aiogram 3.x, используя Dispatcher и фильтр F.
        """

        # @self.dp.message(commands=["upcoming"])
        @self.dp.message(F.text, Command("upcoming"))
        async def cmd_upcoming(message: types.Message):
            text = self.get_deadlines_upcoming_grouped_text()
            await message.answer(text, parse_mode="HTML")
        
        @self.dp.message(F.text, Command("expired"))
        async def cmd_expired(message: types.Message):
            text = self.get_deadlines_expired_grouped_text()
            await message.answer(text, parse_mode="HTML")
            
        @self.dp.message(F.chat.type.in_(["group", "supergroup"]))
        async def handle_new_announcement(message: types.Message):
            """
            Обработчик сообщений из групп/супергрупп, чтобы находить дедлайны.
            """
            chat_id = message.chat.id
            topic_id = message.message_thread_id  # None, если нет топика (просто группа)

            # Проверяем, что этот чат (и, возможно, топик) отслеживается
            if chat_id in self.chats_and_topics:
                topics_list = self.chats_and_topics[chat_id]
                # Если список пуст, значит отслеживаем все топики/сообщения
                # Если не пуст — проверяем, что topic_id присутствует
                if (not topics_list) or (topic_id in topics_list):
                    text = message.text or ""
                    chat_title = message.chat.title or str(chat_id)
                    topic_name = None
                    if topic_id:
                        # Условно считаем, что это имя топика
                        topic_name = f"Topic {topic_id}"
                        # topic_name = f"Topic {message.sender_chat.title}"

                    self.logger.info(
                        f"[GROUP MSG] chat={chat_title}({chat_id}), topic={topic_name}, text={text[:50]}..."
                    )
                    same_topic_deadlines = self.get_deadlines_for_topic(chat_id, topic_id)
                    # Формируем лаконичный список для LLM
                    same_topic_list_str = "\n".join(
                        f"- {d['title']} = {d['deadline_dt'].strftime('%Y-%m-%d')}" for d in same_topic_deadlines
                    )
                    # Анализируем сообщение на дедлайн
                    llm_result = await self.openai_client.analyze_message(text, same_topic_list_str)
                    if llm_result and llm_result.get("has_deadline"):
                        dt_str = llm_result.get("deadline_datetime", "")
                        parsed_dt = self.parse_datetime_from_string(dt_str)
                        if parsed_dt:
                            # Формируем ссылку на сообщение
                            base_id = str(chat_id)[4:] if str(chat_id).startswith("-100") else str(chat_id)
                            msg_link = f"https://t.me/c/{base_id}/{message.message_id}"

                            await self.add_or_update_deadline(
                                chat_id=chat_id,
                                chat_title=chat_title,
                                topic_id=topic_id,
                                topic_name=topic_name,
                                message_id=message.message_id,
                                message_link=msg_link,
                                deadline_title=llm_result.get("deadline_title", "Без названия"),
                                deadline_dt=parsed_dt
                            )
                            self.logger.info("Дедлайн успешно добавлен/обновлён.")
                        else:
                            self.logger.info("Не удалось распарсить дату дедлайна.")
                    else:
                        self.logger.info("В сообщении не найден дедлайн.")

    def get_deadlines_for_topic(self, chat_id: int, topic_id: Optional[int]) -> List[Dict[str,Any]]:
        """
        Возвращает все дедлайны в self.deadlines_storage по (chat_id, topic_id).
        """
        results = []
        for d_data in self.deadlines_storage.values():
            if d_data["chat_id"] == chat_id and d_data["topic_id"] == topic_id:
                results.append(d_data)
        return results

    def parse_datetime_from_string(self, dt_str: str) -> Optional[datetime]:
        """
        Упрощённая функция разбора строки даты/времени.
        Форматы, которые пытаемся распознать:
         - YYYY-MM-DD HH:MM
         - YYYY-MM-DD
        """
        if not dt_str:
            return None
        formats = ["%Y-%m-%d %H:%M", "%Y-%m-%d"]
        for fmt in formats:
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                pass
        return None

    async def add_or_update_deadline(
        self,
        chat_id: int,
        chat_title: str,
        topic_id: Optional[int],
        topic_name: Optional[str],
        message_id: int,
        message_link: str,
        deadline_title: str,
        deadline_dt: datetime
    ):
        """
        Добавляем или обновляем дедлайн в self.deadlines_storage.
        Привязываем к (chat_id, topic_id, title) для упрощённой идентификации.
        """
        existing_id = None
        for d_id, data in self.deadlines_storage.items():
            same_chat = (data["chat_id"] == chat_id)
            same_topic = (data["topic_id"] == topic_id)
            same_title = (data["title"] == deadline_title)
            if same_chat and same_topic and same_title:
                existing_id = d_id
                break

        now_str = self.get_current_time().strftime("%Y-%m-%d %H:%M:%S")
        if existing_id:
            # Обновляем
            self.deadlines_storage[existing_id]["deadline_dt"] = deadline_dt
            self.deadlines_storage[existing_id]["message_id"] = message_id
            self.deadlines_storage[existing_id]["message_link"] = message_link
            self.deadlines_storage[existing_id]["chat_title"] = chat_title
            self.deadlines_storage[existing_id]["topic_name"] = topic_name
            self.deadlines_storage[existing_id]["history"].append(
                (now_str, f"Обновлён дедлайн: {deadline_title}", message_link)
            )
        else:
            new_id = f"deadline_{len(self.deadlines_storage) + 1}"
            self.deadlines_storage[new_id] = {
                "chat_id": chat_id,
                "chat_title": chat_title,
                "topic_id": topic_id,
                "topic_name": topic_name,
                "message_id": message_id,
                "message_link": message_link,
                "title": deadline_title,
                "deadline_dt": deadline_dt,
                "history": [(now_str, f"Создан дедлайн: {deadline_title}", message_link)]
            }

    def get_deadlines_upcoming(self) -> List[dict]:
        """
        Возвращает список дедлайнов, которые ещё не просрочены.
        """
        now = self.get_current_time()
        result = []
        for item in self.deadlines_storage.values():
            if item["deadline_dt"].date() >= now.date():
                result.append(item)
        return sorted(result, key=lambda x: x["deadline_dt"])

    def get_deadlines_expired(self) -> List[dict]:
        """
        Возвращает список просроченных дедлайнов.
        """
        now = self.get_current_time()
        result = []
        for item in self.deadlines_storage.values():
            print(item["deadline_dt"], item["deadline_dt"].date() < now.date())
            if item["deadline_dt"].date() < now.date():
                result.append(item)
        return sorted(result, key=lambda x: x["deadline_dt"])

    def get_deadlines_upcoming_grouped_text(self) -> str:
        upcoming = self.get_deadlines_upcoming()
        if not upcoming:
            return "Нет предстоящих дедлайнов."

        grouped = {}
        for d in upcoming:
            key = (
                d["chat_id"],
                d.get("chat_title"),
                d.get("topic_id"),
                d.get("topic_name"),
            )
            grouped.setdefault(key, []).append(d)

        lines = ['Список дедлайнов:\n']
        for (chat_id, chat_title, topic_id, topic_name), items in grouped.items():
            title_line = f"{chat_title}:"
            lines.append(title_line)
            for d_item in items:
                dt_str = d_item["deadline_dt"].strftime("%Y-%m-%d")
                free_time = (d_item["deadline_dt"].date() - self.get_current_time().date()).days
                line = f" • <i>{d_item['title']}</i> до {dt_str} | Осталось {free_time if free_time else 'меньше 1'} дня(-ей)\n   <a href=\"{d_item['message_link']}\">Ссылка</a>"
                if len(d_item.get("history", [])) > 1:
                    history_links = []
                    # Выводим все предыдущие события (кроме последнего, текущее обновление)
                    for event in d_item["history"][:-1]:
                        event_time, event_desc, event_link = event
                        history_links.append(f"<a href=\"{event_link}\">{event_desc}</a>")
                    line += "\n   История: " + ", ".join(history_links)
                lines.append(line)
        return "\n".join(lines)

    def get_deadlines_expired_grouped_text(self) -> str:
        expired = self.get_deadlines_expired()
        if not expired:
            return "Нет просроченных дедлайнов."

        grouped = {}
        for d in expired:
            key = (
                d["chat_id"],
                d.get("chat_title"),
                d.get("topic_id"),
                d.get("topic_name"),
            )
            grouped.setdefault(key, []).append(d)

        lines = []
        for (chat_id, chat_title, topic_id, topic_name), items in grouped.items():
            title_line = f"\n{chat_title}"
            lines.append(title_line)
            for d_item in items:
                dt_str = d_item["deadline_dt"].strftime("%Y-%m-%d")
                line = f" • <i>{d_item['title']}</i> (просрочен {dt_str})\n   <a href=\"{d_item['message_link']}\">Ссылка</a>"
                if len(d_item.get("history", [])) > 1:
                    history_links = []
                    for event in d_item["history"][:-1]:
                        event_time, event_desc, event_link = event
                        history_links.append(f"<a href=\"{event_link}\">{event_desc}</a>")
                    line += "\n   История: " + ", ".join(history_links)
                lines.append(line)
        return "\n".join(lines)

    async def run_bot(self):
        """
        Запуск асинхронного цикла бота.
        Бот читает сообщения из отслеживаемых чатов.
        """
        self.logger.info("Запуск long-polling...")
        await self.dp.start_polling(self.bot)




    
# -----------------------
# Пример использования
# -----------------------
if __name__ == "__main__":

    from dotenv import load_dotenv
    load_dotenv()

    openai_base_url = os.getenv("AI_BASE_URL", "https://api.openai.com/v1")
    openai_api_key = os.getenv("AI_KEY", "your_api_key_here")
    openai_model  = os.getenv("AI_MODEL", "gpt-3.5-turbo")

    openai_client = OpenAIClient(
        api_key=openai_api_key,
        base_url=openai_base_url,
        model_name=openai_model
    )

    # Пример настроек
    config = {
        "time_offset": timedelta(days=0, hours=2, minutes=40),
        # Указываем чаты и топики (если список пуст, значит все топики в данном чате)
        "chats_and_topics": {
            -None: [None],            # Отслеживать все топики в данной супергруппе
            -None: [None],   
        }
    }
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "your_telegram_bot_token_here")

    bot_instance = TelegramDeadlineBot(
        telegram_token=telegram_token,
        openai_client=openai_client,
        config=config
    )

    # Запускаем бота в асинхронном режиме
    asyncio.run(bot_instance.run_bot())
