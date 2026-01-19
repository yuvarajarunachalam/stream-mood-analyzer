import pandas as pd
import numpy as np
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import json
import hashlib
import re
from config import AZURE_ENDPOINT, AZURE_KEY

class StreamAnalyzer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.client = self.authenticate_azure()
        
        # Gaming context keywords for sentiment adjustment
        self.positive_emotes = ['PogChamp', 'Pog', 'POG', 'LUL', 'KEKW', 'Kreygasm', 'Clap', 
                                'GG', 'EZ', 'W', 'hype', 'lit', 'poggers', 'based']
        self.negative_emotes = ['BibleThump', 'FeelsBadMan', 'Sadge', 'PepeHands', 'L', 
                                'rip', 'oof', 'unlucky', 'pepehands']
        
    def anonymize_username(self, username):
        """Anonymize username using hash"""
        hash_object = hashlib.md5(username.encode())
        return f"User_{hash_object.hexdigest()[:8]}"
    
    def preprocess_message(self, message):
        """Preprocess message to add gaming context hints"""
        msg_lower = message.lower()
        
        # Detect positive gaming terms
        positive_count = sum(1 for term in self.positive_emotes if term.lower() in msg_lower)
        negative_count = sum(1 for term in self.negative_emotes if term.lower() in msg_lower)
        
        # Add context hints for Azure
        context_hint = ""
        if positive_count > negative_count:
            context_hint = " [excited positive reaction]"
        elif negative_count > positive_count:
            context_hint = " [disappointed reaction]"
        
        # Detect excessive caps (excitement)
        caps_ratio = sum(1 for c in message if c.isupper()) / max(len(message), 1)
        if caps_ratio > 0.5 and len(message) > 3:
            context_hint += " [enthusiastic]"
        
        return message + context_hint, positive_count, negative_count
    
    def adjust_sentiment_score(self, sentiment, scores, positive_count, negative_count):
        """Adjust sentiment based on gaming context"""
        adjusted_sentiment = sentiment
        adjusted_scores = scores.copy()
        
        # If gaming emotes detected, boost corresponding sentiment
        if positive_count > 0:
            boost = min(0.15 * positive_count, 0.3)  # Max 30% boost
            adjusted_scores['positive'] = min(adjusted_scores['positive'] + boost, 1.0)
            adjusted_scores['negative'] = max(adjusted_scores['negative'] - boost/2, 0.0)
            adjusted_scores['neutral'] = 1.0 - adjusted_scores['positive'] - adjusted_scores['negative']
            
            if adjusted_scores['positive'] > max(adjusted_scores['neutral'], adjusted_scores['negative']):
                adjusted_sentiment = 'positive'
        
        elif negative_count > 0:
            boost = min(0.15 * negative_count, 0.3)
            adjusted_scores['negative'] = min(adjusted_scores['negative'] + boost, 1.0)
            adjusted_scores['positive'] = max(adjusted_scores['positive'] - boost/2, 0.0)
            adjusted_scores['neutral'] = 1.0 - adjusted_scores['positive'] - adjusted_scores['negative']
            
            if adjusted_scores['negative'] > max(adjusted_scores['neutral'], adjusted_scores['positive']):
                adjusted_sentiment = 'negative'
        
        return adjusted_sentiment, adjusted_scores
        
    def authenticate_azure(self):
        """Authenticate with Azure Text Analytics"""
        credential = AzureKeyCredential(AZURE_KEY)
        client = TextAnalyticsClient(endpoint=AZURE_ENDPOINT, credential=credential)
        print("[SUCCESS] Connected to Azure Text Analytics")
        return client
    
    def load_data(self):
        """Load chat data from CSV"""
        self.df = pd.read_csv(self.csv_file)
        
        # Anonymize usernames
        self.df['original_username'] = self.df['username']  
        self.df['username'] = self.df['username'].apply(self.anonymize_username)
        
        print(f"[SUCCESS] Loaded {len(self.df)} messages")
        print(f"[INFO] Time range: {self.df['elapsed_seconds'].min()}s to {self.df['elapsed_seconds'].max()}s")
        print(f"[INFO] Usernames anonymized for privacy")
        return self.df
    
    def analyze_sentiment_batch(self):
        """Analyze sentiment using Azure Text Analytics in batches"""
        print("\n[PROCESSING] Analyzing sentiment with gaming context awareness...")
        
        sentiments = []
        scores_positive = []
        scores_neutral = []
        scores_negative = []
        gaming_context_detections = []
        
        # Process in batches of 10 (Azure limit)
        batch_size = 10
        total_batches = len(self.df) // batch_size + (1 if len(self.df) % batch_size != 0 else 0)
        
        for i in range(0, len(self.df), batch_size):
            batch_df = self.df.iloc[i:i+batch_size]
            
            # Preprocess messages
            preprocessed_data = [self.preprocess_message(msg) for msg in batch_df['message'].tolist()]
            preprocessed_messages = [item[0] for item in preprocessed_data]
            positive_counts = [item[1] for item in preprocessed_data]
            negative_counts = [item[2] for item in preprocessed_data]
            
            try:
                response = self.client.analyze_sentiment(documents=preprocessed_messages, language="en")
                
                for idx, doc in enumerate(response):
                    if not doc.is_error:
                        # Get Azure's sentiment
                        azure_sentiment = doc.sentiment
                        azure_scores = {
                            'positive': doc.confidence_scores.positive,
                            'neutral': doc.confidence_scores.neutral,
                            'negative': doc.confidence_scores.negative
                        }
                        
                        # Adjust based on gaming context
                        adjusted_sentiment, adjusted_scores = self.adjust_sentiment_score(
                            azure_sentiment, 
                            azure_scores,
                            positive_counts[idx],
                            negative_counts[idx]
                        )
                        
                        sentiments.append(adjusted_sentiment)
                        scores_positive.append(adjusted_scores['positive'])
                        scores_neutral.append(adjusted_scores['neutral'])
                        scores_negative.append(adjusted_scores['negative'])
                        
                        # Track if gaming context was detected
                        gaming_context_detections.append(positive_counts[idx] + negative_counts[idx] > 0)
                    else:
                        sentiments.append("neutral")
                        scores_positive.append(0.33)
                        scores_neutral.append(0.34)
                        scores_negative.append(0.33)
                        gaming_context_detections.append(False)
                
                # Progress update
                current_batch = (i // batch_size) + 1
                if current_batch % 5 == 0 or current_batch == total_batches:
                    print(f"  Processed {current_batch}/{total_batches} batches...")
                    
            except Exception as e:
                print(f"  Error in batch {i}: {e}")
                # Fill with neutral for failed batch
                for _ in range(len(preprocessed_messages)):
                    sentiments.append("neutral")
                    scores_positive.append(0.33)
                    scores_neutral.append(0.34)
                    scores_negative.append(0.33)
                    gaming_context_detections.append(False)
        
        # Add sentiment data to dataframe
        self.df['sentiment'] = sentiments
        self.df['positive_score'] = scores_positive
        self.df['neutral_score'] = scores_neutral
        self.df['negative_score'] = scores_negative
        self.df['gaming_context_detected'] = gaming_context_detections
        
        gaming_adjusted = sum(gaming_context_detections)
        print(f"[SUCCESS] Sentiment analysis complete!")
        print(f"  Positive: {(self.df['sentiment'] == 'positive').sum()} messages")
        print(f"  Neutral: {(self.df['sentiment'] == 'neutral').sum()} messages")
        print(f"  Negative: {(self.df['sentiment'] == 'negative').sum()} messages")
        print(f"  Gaming context detected: {gaming_adjusted} messages")
        
        return self.df
    
    def create_interactive_visualizations(self):
        """Create interactive Plotly visualizations"""
        print("\n[PROCESSING] Creating interactive visualizations...")
        
        # 1. SENTIMENT TIMELINE
        fig_sentiment = go.Figure()
        
        # Group by time windows (30 second intervals)
        self.df['time_window'] = (self.df['elapsed_seconds'] // 30) * 30
        sentiment_timeline = self.df.groupby(['time_window', 'sentiment']).size().unstack(fill_value=0)
        
        colors = {'positive': '#00CC96', 'neutral': '#636EFA', 'negative': '#EF553B'}
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment in sentiment_timeline.columns:
                fig_sentiment.add_trace(go.Scatter(
                    x=sentiment_timeline.index,
                    y=sentiment_timeline[sentiment],
                    mode='lines+markers',
                    name=sentiment.capitalize(),
                    line=dict(color=colors[sentiment], width=2),
                    marker=dict(size=6)
                ))
        
        fig_sentiment.update_layout(
            title="Sentiment Over Time (30s intervals)",
            xaxis_title="Time (seconds)",
            yaxis_title="Message Count",
            hovermode='x unified',
            template='plotly_dark',
            height=500
        )
        fig_sentiment.write_html("sentiment_timeline.html")
        print("  [SUCCESS] Created: sentiment_timeline.html")
        
        # 2. MESSAGE FREQUENCY HEATMAP
        self.df['minute'] = self.df['elapsed_seconds'] // 60
        msg_per_minute = self.df.groupby('minute').size()
        
        fig_frequency = go.Figure(data=go.Bar(
            x=msg_per_minute.index,
            y=msg_per_minute.values,
            marker_color='lightskyblue',
            hovertemplate='Minute %{x}<br>Messages: %{y}<extra></extra>'
        ))
        
        fig_frequency.update_layout(
            title="Message Frequency per Minute",
            xaxis_title="Time (minutes)",
            yaxis_title="Message Count",
            template='plotly_dark',
            height=400
        )
        fig_frequency.write_html("message_frequency.html")
        print("  [SUCCESS] Created: message_frequency.html")
        
        # 3. TOP ACTIVE USERS
        top_users = self.df['username'].value_counts().head(15)
        
        fig_users = go.Figure(data=go.Bar(
            x=top_users.values,
            y=top_users.index,
            orientation='h',
            marker_color='mediumseagreen',
            hovertemplate='%{y}: %{x} messages<extra></extra>'
        ))
        
        fig_users.update_layout(
            title="Top 15 Most Active Users",
            xaxis_title="Message Count",
            yaxis_title="Username",
            template='plotly_dark',
            height=600
        )
        fig_users.write_html("top_users.html")
        print("  [SUCCESS] Created: top_users.html")
        
        # 4. SENTIMENT SCORE DISTRIBUTION
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Histogram(
            x=self.df['positive_score'],
            name='Positive',
            opacity=0.7,
            marker_color='green'
        ))
        fig_dist.add_trace(go.Histogram(
            x=self.df['negative_score'],
            name='Negative',
            opacity=0.7,
            marker_color='red'
        ))
        
        fig_dist.update_layout(
            title="Sentiment Score Distribution",
            xaxis_title="Confidence Score",
            yaxis_title="Frequency",
            barmode='overlay',
            template='plotly_dark',
            height=400
        )
        fig_dist.write_html("sentiment_distribution.html")
        print("  [SUCCESS] Created: sentiment_distribution.html")
        
        # 5. ENGAGEMENT DASHBOARD (Combined)
        fig_dashboard = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Timeline', 'Message Frequency', 
                          'Sentiment Distribution', 'Top Users'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Sentiment timeline
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment in sentiment_timeline.columns:
                fig_dashboard.add_trace(
                    go.Scatter(x=sentiment_timeline.index, y=sentiment_timeline[sentiment],
                             name=sentiment, line=dict(color=colors[sentiment])),
                    row=1, col=1
                )
        
        # Message frequency
        fig_dashboard.add_trace(
            go.Bar(x=msg_per_minute.index, y=msg_per_minute.values, 
                   name='Messages', marker_color='lightskyblue'),
            row=1, col=2
        )
        
        # Sentiment distribution
        fig_dashboard.add_trace(
            go.Bar(x=['Positive', 'Neutral', 'Negative'],
                   y=[self.df['positive_score'].mean(), 
                      self.df['neutral_score'].mean(),
                      self.df['negative_score'].mean()],
                   marker_color=['green', 'gray', 'red']),
            row=2, col=1
        )
        
        # Top users
        top_5 = self.df['username'].value_counts().head(5)
        fig_dashboard.add_trace(
            go.Bar(x=top_5.index, y=top_5.values, marker_color='mediumseagreen'),
            row=2, col=2
        )
        
        fig_dashboard.update_layout(
            title_text="Stream Engagement Dashboard",
            template='plotly_dark',
            height=800,
            showlegend=True
        )
        fig_dashboard.write_html("dashboard.html")
        print("  [SUCCESS] Created: dashboard.html (MAIN DASHBOARD)")
        
    def generate_report(self):
        """Generate statistical report"""
        print("\n[PROCESSING] Generating analysis report...")
        
        report = {
            "recording_summary": {
                "total_messages": len(self.df),
                "duration_seconds": int(self.df['elapsed_seconds'].max()),
                "duration_minutes": round(self.df['elapsed_seconds'].max() / 60, 1),
                "unique_users": self.df['username'].nunique(),
                "avg_messages_per_minute": round(len(self.df) / (self.df['elapsed_seconds'].max() / 60), 2)
            },
            "sentiment_analysis": {
                "positive_messages": int((self.df['sentiment'] == 'positive').sum()),
                "neutral_messages": int((self.df['sentiment'] == 'neutral').sum()),
                "negative_messages": int((self.df['sentiment'] == 'negative').sum()),
                "positive_percentage": round((self.df['sentiment'] == 'positive').sum() / len(self.df) * 100, 2),
                "avg_positive_score": round(self.df['positive_score'].mean(), 3),
                "avg_negative_score": round(self.df['negative_score'].mean(), 3)
            },
            "engagement_metrics": {
                "most_active_user": self.df['username'].value_counts().index[0],
                "most_active_user_messages": int(self.df['username'].value_counts().values[0]),
                "peak_minute": int(self.df.groupby(self.df['elapsed_seconds'] // 60).size().idxmax()),
                "peak_minute_messages": int(self.df.groupby(self.df['elapsed_seconds'] // 60).size().max()),
                "top_5_users": self.df['username'].value_counts().head(5).to_dict()
            }
        }
        
        # Save report
        with open('analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("STREAM ANALYSIS REPORT")
        print("="*50)
        print(f"\n[RECORDING SUMMARY]")
        print(f"  Total Messages: {report['recording_summary']['total_messages']}")
        print(f"  Duration: {report['recording_summary']['duration_minutes']} minutes")
        print(f"  Unique Users: {report['recording_summary']['unique_users']}")
        print(f"  Avg Messages/Min: {report['recording_summary']['avg_messages_per_minute']}")
        
        print(f"\n[SENTIMENT ANALYSIS]")
        print(f"  Positive: {report['sentiment_analysis']['positive_messages']} ({report['sentiment_analysis']['positive_percentage']}%)")
        print(f"  Neutral: {report['sentiment_analysis']['neutral_messages']}")
        print(f"  Negative: {report['sentiment_analysis']['negative_messages']}")
        
        print(f"\n[ENGAGEMENT METRICS]")
        print(f"  Most Active User: {report['engagement_metrics']['most_active_user']} ({report['engagement_metrics']['most_active_user_messages']} messages)")
        print(f"  Peak Activity: Minute {report['engagement_metrics']['peak_minute']} ({report['engagement_metrics']['peak_minute_messages']} messages)")
        
        print("\n[SUCCESS] Full report saved to: analysis_report.json")
        print("="*50)
        
        return report
    
    def save_analyzed_data(self):
        """Save the complete analyzed dataset"""
        output_file = self.csv_file.replace('.csv', '_analyzed.csv')
        
        # Remove original_username column 
        output_df = self.df.drop(columns=['original_username'], errors='ignore')
        output_df.to_csv(output_file, index=False)
        
        print(f"\n[SUCCESS] Analyzed data saved to: {output_file}")
        print(f"[INFO] Original usernames excluded for privacy")

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyzer.py <csv_file>")
        print("Example: python analyzer.py chat_data_20260119_160204.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    print("="*50)
    print("STREAM MOOD & REACTION ANALYZER")
    print("="*50)
    
    # Initialize analyzer
    analyzer = StreamAnalyzer(csv_file)
    
    # Load data
    analyzer.load_data()
    
    # Analyze sentiment
    analyzer.analyze_sentiment_batch()
    
    # Create visualizations
    analyzer.create_interactive_visualizations()
    
    # Generate report
    analyzer.generate_report()
    
    # Save analyzed data
    analyzer.save_analyzed_data()
    
    print("\n[COMPLETE] Analysis finished successfully!")
    print("\n[GENERATED FILES]")
    print("  - dashboard.html (Main interactive dashboard)")
    print("  - sentiment_timeline.html")
    print("  - message_frequency.html")
    print("  - top_users.html")
    print("  - sentiment_distribution.html")
    print("  - analysis_report.json")
    print("  - [filename]_analyzed.csv")
    
    print("\n[INFO] Open dashboard.html in your browser to view results!")

if __name__ == '__main__':
    main()