import socket
import csv
import datetime
import re
from config import TWITCH_ACCESS_TOKEN, TWITCH_CHANNEL

class TwitchChatRecorder:
    def __init__(self, channel, duration_minutes=20):
        self.channel = channel.lower()
        self.token = TWITCH_ACCESS_TOKEN
        self.duration = duration_minutes * 60  # Convert to seconds
        self.messages = []
        self.start_time = None
        
        # IRC connection settings
        self.server = 'irc.chat.twitch.tv'
        self.port = 6667
        self.nickname = 'justinfan12345'  # Anonymous user
        self.sock = None
        
    def connect(self):
        """Connect to Twitch IRC"""
        self.sock = socket.socket()
        self.sock.connect((self.server, self.port))
        
        # Authenticate
        self.sock.send(f"PASS oauth:{self.token}\n".encode('utf-8'))
        self.sock.send(f"NICK {self.nickname}\n".encode('utf-8'))
        self.sock.send(f"JOIN #{self.channel}\n".encode('utf-8'))
        
        print(f'✓ Connected to Twitch IRC')
        print(f'✓ Monitoring channel: {self.channel}')
        print(f'✓ Recording for {self.duration // 60} minutes...')
        print(f'✓ Started at: {datetime.datetime.now().strftime("%H:%M:%S")}\n')
        
    def parse_message(self, raw_message):
        """Parse IRC message to extract username and content"""
        pattern = r':(.+?)!.+?PRIVMSG #.+? :(.+)'
        match = re.match(pattern, raw_message)
        if match:
            username = match.group(1)
            message = match.group(2).strip()
            return username, message
        return None, None
        
    def record(self):
        """Start recording chat messages"""
        self.start_time = datetime.datetime.now()
        self.connect()
        
        while True:
            try:
                # Receive data from socket
                response = self.sock.recv(2048).decode('utf-8', errors='ignore')
                
                # Handle PING to keep connection alive
                if response.startswith('PING'):
                    self.sock.send("PONG\n".encode('utf-8'))
                    continue
                
                # Check if recording duration exceeded
                elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
                if elapsed > self.duration:
                    print(f'\n✓ Recording complete! Recorded {len(self.messages)} messages')
                    self.save_to_csv()
                    break
                
                # Parse messages
                for line in response.split('\r\n'):
                    if 'PRIVMSG' in line:
                        username, message = self.parse_message(line)
                        if username and message:
                            msg_data = {
                                'timestamp': datetime.datetime.now().isoformat(),
                                'username': username,
                                'message': message,
                                'elapsed_seconds': int(elapsed)
                            }
                            self.messages.append(msg_data)
                            
                            # Progress update every 100 messages
                            if len(self.messages) % 100 == 0:
                                print(f'Recorded {len(self.messages)} messages... ({int(elapsed/60)} min {int(elapsed%60)} sec elapsed)')
                
            except Exception as e:
                print(f'Error: {e}')
                break
        
        self.sock.close()
        
    def save_to_csv(self):
        """Save recorded messages to CSV file"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'chat_data_{timestamp}.csv'
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'username', 'message', 'elapsed_seconds'])
            writer.writeheader()
            writer.writerows(self.messages)
        
        print(f'✓ Data saved to: {filename}')
        print(f'✓ Total messages: {len(self.messages)}')

if __name__ == '__main__':
    print('Starting Twitch Chat Recorder...')
    print(f'Target: {TWITCH_CHANNEL}')
    print(f'Duration: 20 minutes\n')
    
    recorder = TwitchChatRecorder(TWITCH_CHANNEL, duration_minutes=20)
    recorder.record()