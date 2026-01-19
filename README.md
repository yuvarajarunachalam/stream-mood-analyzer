# Stream Mood & Reaction Analyzer

A real-time sentiment analysis system for Twitch livestream chat data, built for analyzing viewer engagement patterns and emotional responses during live streams.

## Project Overview

This project captures and analyzes Twitch chat messages to understand stream engagement through sentiment analysis and statistical methods. The system records chat data, processes it using Azure Text Analytics API with gaming-context awareness, and generates interactive visualizations to identify engagement patterns, sentiment trends, and peak moments during streams.

## Features

- Real-time Twitch chat data collection via IRC protocol
- Cloud-based sentiment analysis using Azure Text Analytics API
- Gaming context awareness for improved sentiment accuracy (detects Twitch emotes and gaming slang)
- Username anonymization for privacy protection
- Interactive visualizations using Plotly
- Statistical analysis with time-series insights
- Comprehensive engagement metrics and reporting

## Technology Stack

### Core Technologies
- Python 3.8+
- Azure Text Analytics API (Cognitive Services)
- Twitch IRC protocol for chat access

### Libraries
- pandas - Data manipulation and analysis
- plotly - Interactive data visualizations
- azure-ai-textanalytics - Azure Cognitive Services integration
- numpy - Numerical computing
- socket - IRC connection handling

## Architecture

### Data Collection Layer
The recorder module connects to Twitch IRC servers and captures live chat messages with timestamps. Messages are stored with metadata including username, content, and elapsed time.

### Analysis Layer
The analyzer module performs:
1. Username anonymization using MD5 hashing
2. Gaming context preprocessing (emote and slang detection)
3. Sentiment scoring via Azure Text Analytics
4. Context-aware sentiment adjustment
5. Statistical aggregation and pattern detection

### Visualization Layer
Interactive HTML dashboards are generated showing:
- Sentiment timeline with 30-second intervals
- Message frequency distribution
- Top active users (anonymized)
- Sentiment score distributions
- Combined engagement dashboard

## Installation

### Prerequisites
- Python 3.8 or higher
- Azure account with Text Analytics resource
- Twitch account

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stream-mood-analyzer.git
cd stream-mood-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure credentials:
Create a `config.py` file with your credentials:
```python
TWITCH_ACCESS_TOKEN = "your_token_here"
TWITCH_CHANNEL = "channel_name"
AZURE_ENDPOINT = "your_azure_endpoint"
AZURE_KEY = "your_azure_key"
```

Note: Never commit `config.py` to version control.

## Usage

### Recording Chat Data

```bash
python recorder.py
```

This will record chat messages for 20 minutes and save to a timestamped CSV file.

### Analyzing Recorded Data

```bash
python analyzer.py chat_data_TIMESTAMP.csv
```

This generates:
- Interactive HTML visualizations
- Statistical analysis report (JSON)
- Analyzed dataset with sentiment scores (CSV)

### Viewing Results

Open `dashboard.html` in your web browser to view the interactive engagement dashboard.

## Project Structure

```
stream-mood-analyzer/
├── recorder.py              # Twitch chat recording module
├── analyzer.py              # Sentiment analysis and visualization
├── requirements.txt         # Python dependencies
├── config.py               # Configuration (not tracked)
├── .gitignore              # Git ignore rules
├── README.md               # Project documentation
└── LICENSE                 # MIT License
```

## Methodology

### Sentiment Analysis Approach

The system employs a hybrid approach combining cloud-based NLP with domain-specific adjustments:

1. Message preprocessing detects gaming-specific terms (POG, GG, Sadge, etc.)
2. Context hints are added to messages before Azure analysis
3. Azure Text Analytics provides base sentiment scores
4. Scores are adjusted based on detected gaming context
5. Final sentiment classification considers both Azure output and gaming terminology

This approach improves accuracy for gaming-specific content while maintaining cloud-based processing efficiency.

### Privacy Considerations

All usernames are anonymized using MD5 hashing before visualization or export. The anonymization is irreversible and consistent, allowing for user tracking within analysis while protecting identity.

## Results and Insights

The system successfully:
- Captured 634 messages over 20 minutes from a live stream
- Identified sentiment patterns correlating with gameplay events
- Detected engagement spikes during key moments
- Provided statistical metrics on viewer participation

Sentiment distribution showed contextual awareness improvements with gaming terminology detection active on a significant portion of messages.

## Future Work and Limitations

### Current Limitations
- 20-minute recording window limits pattern detection
- Generic sentiment model has limitations with sarcasm
- Single stream analysis lacks comparative insights
- Azure free tier constrains batch processing speed

### Proposed Enhancements
1. Extended recording periods (multiple hours) for deeper pattern analysis
2. Multi-stream comparative analysis to identify platform-wide trends
3. Correlation analysis between in-game events and chat sentiment
4. User behavior clustering to identify viewer archetypes
5. Real-time dashboard for live sentiment monitoring
6. Integration of emote-only message analysis
7. Language-specific sentiment models for international streams
8. Temporal pattern recognition across multiple stream sessions

### Research Directions
- Investigate correlation between sentiment trends and viewer retention
- Analyze sentiment impact on streamer performance metrics
- Develop predictive models for engagement forecasting
- Explore real-time sentiment as feedback mechanism for streamers

## Academic Context

This project was developed as a final year data science project at VIT Chennai, demonstrating practical applications of:
- Cloud computing services (Azure)
- Natural language processing
- Real-time data streaming
- Statistical analysis and visualization
- API integration and data engineering

The work represents a preliminary exploration of livestream engagement analysis, with significant opportunities for expansion into production-grade systems or deeper research applications.

## Contributing

This is an academic project, but contributions and suggestions are welcome. Please open an issue to discuss proposed changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- VIT Chennai for academic support
- Azure for Education credits
- Twitch for IRC API access
- Anthropic Claude for development assistance

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Note:** This is a preliminary research project. Production deployment would require additional error handling, scalability improvements, and compliance considerations.
