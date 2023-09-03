from loguru import logger


def main():
    logger.info('Logger has been started ...')


if __name__ == '__main__':
    logger.add("logs/{time:YYYY-MM-DD-HH-MM}.log", rotation="3 hours", retention="30 days")
    main()
