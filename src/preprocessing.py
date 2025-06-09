import pandas as pd
import glob, csv, json

def mergecsv():
    # Đường dẫn đến thư mục chứa các file CSV
    folder_path = r'C:\Users\laptop\PycharmProjects\pythonProject/TTHC*.csv'

    # Lấy danh sách tất cả các file CSV trong thư mục
    all_files = glob.glob(folder_path)

    # Đọc và gộp tất cả các file lại
    df_list = [pd.read_csv(f) for f in all_files]
    merged_df = pd.concat(df_list, ignore_index=True)

    # Lưu file gộp ra CSV mới
    merged_df.to_csv('TTHC.csv', index=False, encoding='utf-8-sig')
    print("Đã gộp các file CSV thành công và lưu vào TTHC.csv")

def fix():
    def fix_ma_thu_tuc(ma):
        try:
            if pd.isna(ma):
                return ma
            ma = str(ma).strip()
            parts = ma.split('.')
            if len(parts) != 2:
                return ma  # Không đúng định dạng, giữ nguyên

            phan_truoc, phan_sau = parts

            # Nếu phần sau là số và ngắn hơn 6 ký tự → thêm 0 vào cuối
            if phan_sau.isdigit() and len(phan_sau) < 6:
                phan_sau = phan_sau.ljust(6, '0')

            return f"{phan_truoc}.{phan_sau}"
        except:
            return ma

    input_file = "TTHC.csv"
    output_file = "fixed_TTHC.csv"

    # Đọc toàn bộ các cột dưới dạng chuỗi để tránh mất số 0
    df = pd.read_csv(input_file, dtype=str)

    df['Mã thủ tục'] = df['Mã thủ tục'].apply(fix_ma_thu_tuc)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Đã sửa và lưu vào: {output_file}")

def convert():
    # Đường dẫn file CSV đầu vào và JSON đầu ra
    input_csv = "fixed_TTHC.csv"
    output_json = "TTHC.json"

    # Đọc file CSV và chuyển đổi
    result = []
    with open(input_csv, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            title = row["Tên thủ tục"]
            source = row["Cơ quan thực hiện"]
            url = f"procedure_{idx}"

            # Tạo nội dung gộp từ các trường
            content_parts = []
            for key, value in row.items():
                if key != "Tên thủ tục":  # tránh lặp lại title
                    content_parts.append(f"{key}: {value}")
            content = f"Mã thủ tục: {row['Mã thủ tục']}\n" + "\n".join(content_parts)

            result.append({
                "title": title,
                "content": content,
                "source": source,
                "url": url
            })

    # Ghi ra file JSON
    with open(output_json, mode='w', encoding='utf-8') as jsonfile:
        json.dump(result, jsonfile, ensure_ascii=False, indent=2)

    print(f"Đã chuyển đổi xong. Kết quả lưu vào {output_json}")

mergecsv()
fix()
convert()