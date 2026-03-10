#!/usr/bin/env node

console.error('Avito browser automation disabled: safe mode is enabled.');
console.error('Use manual flow:');
console.error('py job_hunter_assistant.py reply-chat --incoming "..." --profile "..." --employer "..." --vacancy "..." --notify-telegram');
process.exit(1);
