import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os
from matplotlib.dates import DateFormatter
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy import signal
import argparse


def load_and_prepare_data(file_path):
    """Load and prepare power quality data for analysis"""
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    
    # Try to detect timestamp column automatically
    df['Timestamp'] = pd.to_datetime(df.iloc[:, 0], errors='coerce', dayfirst=True)
    original_rows = len(df)
    df = df.dropna(subset=['Timestamp'])
    if len(df) < original_rows:
        print(f"Dropped {original_rows - len(df)} rows with invalid timestamps")
    
    df.set_index('Timestamp', inplace=True)
    return df


def identify_parameter_types(df):
    """Identify different types of power quality parameters"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Categorize columns by parameter type
    voltage_cols = [col for col in numeric_cols if any(x in col.lower() for x in ['v', 'volt', 'voltage'])]
    current_cols = [col for col in numeric_cols if any(x in col.lower() for x in ['i', 'amp', 'current', 'irms'])]
    power_cols = [col for col in numeric_cols if any(x in col.lower() for x in ['power', 'watt', 'va', 'var'])]
    energy_cols = [col for col in numeric_cols if 'energy' in col.lower()]
    thd_cols = [col for col in numeric_cols if any(x in col.lower() for x in ['thd', 'harmonic', 'distortion'])]
    pf_cols = [col for col in numeric_cols if any(x in col.lower() for x in ['pf', 'power factor'])]
    freq_cols = [col for col in numeric_cols if any(x in col.lower() for x in ['freq', 'hz', 'frequency'])]
    
    return {
        'voltage': voltage_cols,
        'current': current_cols,
        'power': power_cols,
        'energy': energy_cols,
        'thd': thd_cols,
        'pf': pf_cols,
        'frequency': freq_cols,
        'all': numeric_cols
    }


def create_dashboard(df, param_types, output_dir="data"):
    """Create a comprehensive power quality dashboard"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a multi-panel figure
    plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(6, 2, height_ratios=[1, 1, 1, 1, 1, 2])
    
    # 1. Voltage Profile
    if param_types['voltage']:
        ax1 = plt.subplot(gs[0, 0])
        for col in param_types['voltage'][:3]:  # Limit to 3 for readability
            df[col].plot(ax=ax1, label=col)
        ax1.set_title('Voltage Profile')
        ax1.set_ylabel('Voltage')
        ax1.legend(loc='best')
        ax1.grid(True)
    
    # 2. Current Profile
    if param_types['current']:
        ax2 = plt.subplot(gs[0, 1])
        for col in param_types['current'][:3]:  # Limit to 3 for readability
            df[col].plot(ax=ax2, label=col)
        ax2.set_title('Current Profile')
        ax2.set_ylabel('Current')
        ax2.legend(loc='best')
        ax2.grid(True)
    
    # 3. Power Factor
    if param_types['pf']:
        ax3 = plt.subplot(gs[1, 0])
        for col in param_types['pf'][:3]:  # Limit to 3 for readability
            df[col].plot(ax=ax3, label=col)
        ax3.set_title('Power Factor')
        ax3.set_ylabel('PF')
        ax3.axhline(y=0.9, color='r', linestyle='--', label='PF Threshold (0.9)')
        ax3.legend(loc='best')
        ax3.grid(True)
    
    # 4. THD Analysis
    if param_types['thd']:
        ax4 = plt.subplot(gs[1, 1])
        for col in param_types['thd'][:3]:  # Limit to 3 for readability
            df[col].plot(ax=ax4, label=col)
        ax4.set_title('Harmonic Distortion')
        ax4.set_ylabel('THD %')
        ax4.axhline(y=5, color='r', linestyle='--', label='THD Threshold (5%)')
        ax4.legend(loc='best')
        ax4.grid(True)
    
    # 5. Power Analysis
    if param_types['power']:
        ax5 = plt.subplot(gs[2, 0])
        for col in param_types['power'][:3]:  # Limit to 3 for readability
            df[col].plot(ax=ax5, label=col)
        ax5.set_title('Power Analysis')
        ax5.set_ylabel('Power')
        ax5.legend(loc='best')
        ax5.grid(True)
    
    # 6. Energy Consumption
    if param_types['energy']:
        ax6 = plt.subplot(gs[2, 1])
        for col in param_types['energy'][:3]:  # Limit to 3 for readability
            df[col].plot(ax=ax6, label=col)
        ax6.set_title('Energy Consumption')
        ax6.set_ylabel('Energy')
        ax6.legend(loc='best')
        ax6.grid(True)
    
    # 7. Daily Profile (using resampling)
    ax7 = plt.subplot(gs[3, 0:2])
    if param_types['power']:
        # Create daily profile for power
        daily_profile = df[param_types['power'][0]].resample('H').mean()
        daily_profile.groupby(daily_profile.index.hour).mean().plot(kind='bar', ax=ax7)
        ax7.set_title('Average Hourly Power Profile')
        ax7.set_xlabel('Hour of Day')
        ax7.set_ylabel('Average Power')
        ax7.grid(True)
    
    # 8. Event Heatmap (time of day vs day of week)
    ax8 = plt.subplot(gs[4, 0:2])
    if param_types['voltage']:
        # Create a heatmap of voltage variations by hour and day
        voltage_col = param_types['voltage'][0]
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        heatmap_data = df.pivot_table(
            values=voltage_col, 
            index='day_of_week', 
            columns='hour', 
            aggfunc=np.std
        )
        sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax8)
        ax8.set_title('Voltage Variation by Hour and Day')
        ax8.set_xlabel('Hour of Day')
        ax8.set_ylabel('Day of Week (0=Monday)')
    
    # 9. Correlation Matrix
    ax9 = plt.subplot(gs[5, 0:2])
    # Select a subset of columns for better readability
    selected_cols = []
    for category in ['voltage', 'current', 'power', 'thd', 'pf']:
        if param_types[category] and len(param_types[category]) > 0:
            selected_cols.append(param_types[category][0])
    
    if selected_cols:
        corr_matrix = df[selected_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax9)
        ax9.set_title('Parameter Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/power_quality_dashboard.png", dpi=300, bbox_inches='tight')
    print(f"Dashboard saved to {output_dir}/power_quality_dashboard.png")
    plt.close()


def detect_events(df, param_types):
    """Detect power quality events based on thresholds and statistical analysis"""
    events = []
    
    # 1. Voltage Sags/Swells
    if param_types['voltage']:
        for col in param_types['voltage']:
            nominal = df[col].median()  # Use median as nominal value
            sag_threshold = 0.9 * nominal
            swell_threshold = 1.1 * nominal
            
            # Detect sags
            sags = df[df[col] < sag_threshold]
            if not sags.empty:
                events.append({
                    'event_type': 'Voltage Sag',
                    'parameter': col,
                    'count': len(sags),
                    'avg_magnitude': sags[col].mean() / nominal * 100,
                    'worst_case': sags[col].min() / nominal * 100,
                    'timestamps': sags.index
                })
            
            # Detect swells
            swells = df[df[col] > swell_threshold]
            if not swells.empty:
                events.append({
                    'event_type': 'Voltage Swell',
                    'parameter': col,
                    'count': len(swells),
                    'avg_magnitude': swells[col].mean() / nominal * 100,
                    'worst_case': swells[col].max() / nominal * 100,
                    'timestamps': swells.index
                })
    
    # 2. THD Violations
    if param_types['thd']:
        for col in param_types['thd']:
            threshold = 5.0  # IEEE 519 standard often uses 5% as threshold
            violations = df[df[col] > threshold]
            if not violations.empty:
                events.append({
                    'event_type': 'High THD',
                    'parameter': col,
                    'count': len(violations),
                    'avg_magnitude': violations[col].mean(),
                    'worst_case': violations[col].max(),
                    'timestamps': violations.index
                })
    
    # 3. Power Factor Violations
    if param_types['pf']:
        for col in param_types['pf']:
            threshold = 0.9
            violations = df[df[col] < threshold]
            if not violations.empty:
                events.append({
                    'event_type': 'Low Power Factor',
                    'parameter': col,
                    'count': len(violations),
                    'avg_magnitude': violations[col].mean(),
                    'worst_case': violations[col].min(),
                    'timestamps': violations.index
                })
    
    # 4. Current Spikes
    if param_types['current']:
        for col in param_types['current']:
            # Use statistical approach to identify spikes
            mean = df[col].mean()
            std = df[col].std()
            threshold = mean + 3 * std  # 3 sigma rule
            
            spikes = df[df[col] > threshold]
            if not spikes.empty:
                events.append({
                    'event_type': 'Current Spike',
                    'parameter': col,
                    'count': len(spikes),
                    'avg_magnitude': spikes[col].mean(),
                    'worst_case': spikes[col].max(),
                    'timestamps': spikes.index
                })
    
    return events


def generate_advanced_report(df, events, param_types, output_dir="data"):
    """Generate an advanced power quality report"""
    os.makedirs(output_dir, exist_ok=True)
    report_path = f"{output_dir}/advanced_power_quality_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("🔍 ADVANCED POWER QUALITY ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Dataset information
        f.write(f"📊 Dataset Information:\n")
        f.write(f"  - Time period: {df.index.min()} to {df.index.max()}\n")
        f.write(f"  - Total records: {len(df)}\n")
        f.write(f"  - Sampling interval: {(df.index[1] - df.index[0]).total_seconds()} seconds\n\n")
        
        # Parameter summary
        f.write(f"📈 Parameter Summary:\n")
        for category, cols in param_types.items():
            if category != 'all' and cols:
                f.write(f"  - {category.capitalize()} parameters: {len(cols)}\n")
        f.write("\n")
        
        # Event summary
        if events:
            f.write(f"⚠️ Power Quality Events Summary:\n")
            event_types = {}
            for event in events:
                event_key = f"{event['event_type']} in {event['parameter']}"
                event_types[event_key] = event['count']
            
            # Sort by count in descending order
            for event_key, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  - {event_key}: {count} occurrences\n")
            f.write("\n")
            
            # Detailed event analysis
            f.write(f"🔎 Detailed Event Analysis:\n")
            for event in events:
                f.write(f"\n  {event['event_type']} in {event['parameter']}:\n")
                f.write(f"    - Occurrences: {event['count']}\n")
                f.write(f"    - Average magnitude: {event['avg_magnitude']:.2f}\n")
                f.write(f"    - Worst case: {event['worst_case']:.2f}\n")
                
                # Show a few example timestamps
                if len(event['timestamps']) > 0:
                    f.write(f"    - Example occurrences:\n")
                    for ts in event['timestamps'][:3]:  # Show first 3 examples
                        f.write(f"      * {ts}\n")
                    if len(event['timestamps']) > 3:
                        f.write(f"      * ... and {len(event['timestamps'])-3} more\n")
        else:
            f.write("⚠️ No significant power quality events detected.\n\n")
        
        # Statistical analysis
        f.write("\n📊 Statistical Analysis:\n")
        
        # Analyze key parameters from each category
        for category, cols in param_types.items():
            if category != 'all' and cols:
                f.write(f"\n  {category.capitalize()} Parameters:\n")
                for col in cols[:3]:  # Limit to first 3 parameters per category
                    stats = df[col].describe()
                    f.write(f"    - {col}:\n")
                    f.write(f"      * Mean: {stats['mean']:.2f}\n")
                    f.write(f"      * Std Dev: {stats['std']:.2f}\n")
                    f.write(f"      * Min: {stats['min']:.2f}\n")
                    f.write(f"      * Max: {stats['max']:.2f}\n")
                    f.write(f"      * Variation: {stats['std']/stats['mean']*100:.2f}%\n")
        
        # Recommendations
        f.write("\n💡 Recommendations:\n")
        
        # Voltage recommendations
        if param_types['voltage']:
            voltage_issues = [e for e in events if e['parameter'] in param_types['voltage']]
            if voltage_issues:
                f.write("\n  1. Voltage Quality Recommendations:\n")
                f.write("    - Monitor voltage variations more closely\n")
                f.write("    - Consider voltage stabilization equipment\n")
                f.write("    - Investigate potential causes: load switching, weak grid, etc.\n")
        
        # Add more recommendations based on other parameters
        f.write("\n  5. General Recommendations:\n")
        f.write("    - Implement continuous power quality monitoring\n")
        f.write("    - Set up alerts for anomalies based on identified thresholds\n")
        f.write("    - Conduct periodic load analysis to identify changes in consumption patterns\n")
        
        f.write("\nReport generated by Advanced Power Quality Analysis Tool\n")
    
    print(f"Advanced report saved to {report_path}")
    return report_path


def main():
    """Main function to run the advanced power quality analysis"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Advanced Power Quality Analysis')
    parser.add_argument('--file', type=str, help='Path to the CSV file containing power quality data')
    args = parser.parse_args()
    
    print("\n===== ADVANCED POWER QUALITY ANALYSIS =====\n")
    
    # Use provided file path or default
    file_path = args.file if args.file else "ES.133 (SN 33030973)_250319_1158_trend.csv"
    
    # Load and prepare data
    df = load_and_prepare_data(file_path)
    
    # Identify parameter types
    param_types = identify_parameter_types(df)
    
    # Print detected parameters
    print("\nDetected parameters:")
    for category, cols in param_types.items():
        if category != 'all' and cols:
            print(f"  - {category.capitalize()}: {len(cols)} parameters")
    
    # Create dashboard
    print("\nGenerating power quality dashboard...")
    create_dashboard(df, param_types)
    
    # Detect events
    print("\nDetecting power quality events...")
    events = detect_events(df, param_types)
    
    # Generate report
    print("\nGenerating advanced power quality report...")
    report_path = generate_advanced_report(df, events, param_types)
    
    print("\n===== ADVANCED POWER QUALITY ANALYSIS COMPLETE =====\n")
    print(f"Results saved to the 'data' directory")


if __name__ == "__main__":
    main()