import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

def generate_darf_excel_report(
    monthly_tax_summaries: List[Dict[str, Any]],
    output_dir: Path
) -> Path:
    """
    Generates an Excel report summarizing monthly tax liabilities.

    Args:
        monthly_tax_summaries: A list of dictionaries, where each dict is the
                               output from `calculate_monthly_taxes`.
        output_dir: The directory where the report will be saved.

    Returns:
        The path to the generated Excel file.
    """
    report_data = []
    for summary in monthly_tax_summaries:
        if not summary or 'details' not in summary:
            continue
            
        month_str = f"{summary['year']}-{summary['month']:02d}"
        
        for asset_class, details in summary['details'].items():
            if isinstance(details, dict) and details.get('gross_sales', 0) > 0:
                report_data.append({
                    'Month': month_str,
                    'Asset Class': asset_class,
                    'Gross Sales (R$)': details.get('gross_sales', 0),
                    'Net Gain/Loss (R$)': details.get('net_gain', 0),
                    'Tax Due (R$)': details.get('tax_due', 0),
                    'DARF Code': '6015' if details.get('tax_due', 0) > 0 else 'N/A'
                })

    if not report_data:
        print("No taxable events to report. DARF report will not be generated.")
        return None

    report_df = pd.DataFrame(report_data)
    
    # --- Formatting the Excel Output ---
    output_path = output_dir / "darf_tax_report.xlsx"
    writer = pd.ExcelWriter(output_path, engine='openpyxl')
    report_df.to_excel(writer, sheet_name='Monthly Tax Summary', index=False)
    
    # Auto-adjust column widths for readability
    worksheet = writer.sheets['Monthly Tax Summary']
    for column in worksheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        if report_df[column[0].value].dtype == 'object':
            max_length = report_df[column[0].value].astype(str).map(len).max()
        else:
            max_length = len(str(column[0].value))
        
        adjusted_width = (max_length + 2)
        worksheet.column_dimensions[column_letter].width = adjusted_width

    writer.close()
    
    print(f"Successfully generated DARF tax report at: {output_path}")
    return output_path


if __name__ == '__main__':
    print("--- Running DARF Reporter Module Standalone Test ---")
    
    # --- GIVEN ---
    # Dummy data mirroring the output of the tax_tracker
    dummy_tax_data = [
        {
            'year': 2023, 'month': 3, 'total_tax_due': 855.0,
            'details': {
                'STOCK': {'gross_sales': 43000.0, 'net_gain': 5500.0, 'tax_due': 825.0},
                'BDR': {'gross_sales': 3200.0, 'net_gain': 200.0, 'tax_due': 30.0},
                'ETF': {'gross_sales': 1050.0, 'net_gain': -50.0, 'tax_due': 0.0}
            }
        },
        {'year': 2023, 'month': 4, 'total_tax_due': 0.0, 'details': 'No sales in this month.'}
    ]
    
    temp_dir = Path("./temp_darf_report")
    temp_dir.mkdir(exist_ok=True)
    
    # --- WHEN ---
    try:
        report_path = generate_darf_excel_report(dummy_tax_data, temp_dir)
        
        # --- THEN ---
        assert report_path is not None, "Report path should not be None."
        assert report_path.exists(), "Excel report file was not created."
        
        # Read back the data to verify content
        df_read = pd.read_excel(report_path)
        
        assert df_read.shape == (3, 6), "Report has incorrect shape."
        assert df_read['Tax Due (R$)'].sum() == 855.0, "Tax amounts in report are incorrect."
        assert df_read[df_read['Asset Class'] == 'ETF']['Tax Due (R$)'].iloc[0] == 0.0
        
        print(f"\nSuccessfully created and validated the report: {report_path}")

    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir) 