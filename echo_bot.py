import os
import logging
import asyncio
from aiogram import Bot, Dispatcher, types, F
from dotenv import load_dotenv



load_dotenv()

# Укажите свой токен бота
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "your_telegram_bot_token_here")


# Укажите ваш Telegram ID (узнать можно через @userinfobot)
YOUR_TELEGRAM_ID = None
assert YOUR_TELEGRAM_ID!=None, 'укажи свой Telegram ID (узнать можно через @userinfobot)'


logging.basicConfig(level=logging.INFO)

bot = Bot(token=TOKEN)
dp = Dispatcher()

@dp.message(F.chat.type.in_(["group", "supergroup"]))
async def forward_group_messages(message: types.Message):
    """Пересылает текстовые сообщения из групп и супергрупп в личные сообщения владельцу бота."""
    chat_id = message.chat.id
    message_id = message.message_id
    id_top= message.message_thread_id
    print('111',chat_id, id_top)

    if message.chat.type == "supergroup":
        # Для супергрупп Telegram требует особый формат ссылки:
        # удаляем префикс "-100" из идентификатора чата
        message_link = f"https://t.me/c/{str(chat_id)[4:]}/{message_id}"
        text = (
            f"<b>Сообщение из {message.chat.title}:</b>\n\n"
            f"{message.text}\n\n"
            f"🔗 <a href='{message_link}'>Ссылка на сообщение</a>"
            f"\n{chat_id}"
            f"\n{id_top}"
        )
    else:
        text = f"<b>Сообщение из {message.chat.title}:</b>\n\n{message.text}, {chat_id}"

    await bot.send_message(YOUR_TELEGRAM_ID, text, parse_mode='HTML')

async def main():
    """Запуск бота."""
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
