import pandas as pd
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ExcelChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("Fishing.xlsx"):
            print("\nExcel file updated, reloading...\n")
            read_excel()

def read_excel():
    try:
        df = pd.read_excel('Fishing.xlsx', usecols="A:C", header=0)
        df = df.dropna(how='all')  # Remove rows where all values are NaN
        print(df)
    except Exception as e:
        print(f"Error reading Excel file: {e}")


if __name__ == "__main__":
    event_handler = ExcelChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=False)
    observer.start()

    print("Watching for changes to Fishing.xlsx... Press Ctrl+C to stop.\n")
    read_excel()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
