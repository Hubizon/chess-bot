.PHONY: all

all: run_adapter ./dist/run_bot/run_bot

run_adapter: run_adapter.cpp
	g++ run_adapter.cpp -o run_adapter

./dist/run_bot/run_bot: run_bot.py
	pyinstaller run_bot.py
