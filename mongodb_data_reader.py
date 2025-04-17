import pymongo
import pandas as pd
from datetime import datetime, timedelta
import os

def connect_to_mongodb():
    """Connect to MongoDB running on localhost"""
    try:
        # Connect to MongoDB server running on localhost
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        
        # Access the power-meter-log database
        db = client["power-meter-log"]
        
        # Access the collection (using the path notation)
        collection = db["datalog/power"]
        
        print("Successfully connected to MongoDB")
        return client, db, collection
    except pymongo.errors.ConnectionFailure as e:
        print(f"Failed to connect to MongoDB: {e}")
        return None, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None

def fetch_power_data(collection, query=None, limit=None):
    """Fetch power data from MongoDB collection with optional filtering"""
    if query is None:
        query = {}
    
    try:
        # Fetch data from collection with optional limit
        if limit:
            cursor = collection.find(query).limit(limit)
        else:
            cursor = collection.find(query)
        
        # Convert to list of dictionaries
        data = list(cursor)
        print(f"Retrieved {len(data)} documents from MongoDB")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def convert_to_dataframe(data):
    """Convert MongoDB data to pandas DataFrame"""
    if not data:
        print("No data to convert to DataFrame")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # Check if 'timestamp' field exists and convert to datetime
    if 'timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('Timestamp', inplace=True)
    
    return df

def save_to_csv(df, output_dir="data"):
    """Save DataFrame to CSV file"""
    if df.empty:
        print("DataFrame is empty, nothing to save")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"{output_dir}/power_data_{timestamp}.csv"
    
    df.to_csv(file_path)
    print(f"Data saved to {file_path}")
    return file_path

def main():
    """Main function to read data from MongoDB and process it"""
    print("\n===== MONGODB POWER DATA READER =====\n")
    
    # Connect to MongoDB
    client, db, collection = connect_to_mongodb()
    
    if collection:
        # Example: Fetch data from the last 24 hours
        # query = {"timestamp": {"$gte": datetime.now() - timedelta(days=1)}}
        
        # For initial testing, fetch a limited number of documents
        data = fetch_power_data(collection, limit=1000)
        
        # Convert to DataFrame
        df = convert_to_dataframe(data)
        
        if not df.empty:
            print("\nData preview:")
            print(df.head())
            
            # Print column information
            print("\nColumns in the data:")
            for col in df.columns:
                print(f"  - {col}")
            
            # Save to CSV
            csv_path = save_to_csv(df)
            
            # You can now use this data with your advanced_power_analysis.py
            print("\nYou can now use this CSV file with your power analysis tools:")
            print(f"  python advanced_power_analysis.py --file {csv_path}")
        
        # Close MongoDB connection
        client.close()
    
    print("\n===== MONGODB POWER DATA READER COMPLETE =====\n")

if __name__ == "__main__":
    main()