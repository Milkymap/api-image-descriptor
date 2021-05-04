from sys import stdout 
from loguru import logger

logger_format = [
    '<W><k>{time: YYYY-MM-DD hh:mm:ss}</k></W>',
    '<e><i>{process.name: <7}</i></e>',
    '<y>{file: <7}</y>',
    '<g>{line: 03d}</g>', 
    '<E><r>{level: ^11}</r></E>', 
    '<Y><k><i>... [{message}] ... </i></k></Y>'
]
logger_separator = ' # '
logger.remove()
logger.add(
    sink=stdout,
    level='TRACE',
    format=logger_separator.join(logger_format)
)