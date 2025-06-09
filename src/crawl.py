from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv

# Khởi tạo trình duyệt
options = webdriver.ChromeOptions()
options.add_argument('--start-maximized')
driver = webdriver.Chrome(options=options)

# Truy cập trang chủ
driver.get("https://dichvucong.gov.vn/p/home/dvc-trang-chu.html")

# Mở tab "Thủ tục hành chính"
WebDriverWait(driver, 20).until(
    EC.element_to_be_clickable((By.LINK_TEXT, "Thủ tục hành chính"))
).click()

# Nhấn "Tìm kiếm nâng cao"
WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "/html/body/cache/div/div/div/div/div[1]/div[1]/div"))).click()

# Chọn cơ quan cần tìm kiếm
dropdown_co_quan_thuc_hien = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "/html/body/cache/div/div/div/div/div[1]/div[2]/div[2]/div[1]/div")))
dropdown_co_quan_thuc_hien.click()

WebDriverWait(driver, 20).until(
    EC.element_to_be_clickable(
        (By.XPATH, "/html/body/cache/div/div/div/div/div[1]/div[2]/div[2]/div[2]/div/select/option[25]")) # Chọn option tương ứng với cơ quan
).click()

# Nhấn "Tìm kiếm"
driver.find_element(By.XPATH, "/html/body/cache/div/div/div/div/div[1]/div[1]/button/span").click()

with open("TTHC_VPTUD.csv", "w", newline='', encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Mã thủ tục", "Số quyết định", "Tên thủ tục", "Cấp thực hiện", "Loại thủ tục", "Lĩnh vực",
        "Trình tự thực hiện", "Đối tượng thực hiện", "Cơ quan thực hiện", "Cơ quan có thẩm quyền",
        "Địa chỉ tiếp nhận hồ sơ", "Cơ quan được ủy quyền", "Cơ quan phối hợp", "Kêt quả thực hiện",
        "Yêu cầu, điều kiện thực hiện", "Mô tả"])

    page = 1
    while True:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "/html/body/cache/div/div/div/div/div[1]/div[5]")))

        rows = driver.find_elements(By.XPATH, "/html/body/cache/div/div/div/div/div[1]/div[5]/table/tbody/tr")
        for i in range(len(rows)):
            try:
                rows = driver.find_elements(By.XPATH, "/html/body/cache/div/div/div/div/div[1]/div[5]/table/tbody/tr")
                ma_thu_tuc_el = rows[i].find_element(By.XPATH, "./td[1]/a")
                link = ma_thu_tuc_el.get_attribute("href")

                # Lưu tab hiện tại và mở chi tiết thủ tục trong tab mới
                main_tab = driver.current_window_handle
                driver.execute_script("window.open(arguments[0]);", link)
                driver.switch_to.window(driver.window_handles[-1])

                # Nhấn vào chi tiết thủ tục
                WebDriverWait(driver, 20).until(
                    EC.element_to_be_clickable((By.XPATH, "/html/body/div[5]/div[1]/div/div/div[1]/div[1]/div[1]/a"))
                ).click()

                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.XPATH, "/html/body/div[5]/div[2]")))


                def get_text_by_label(label_text):
                    try:
                        xpath = f"//div[contains(text(), '{label_text}')]/following-sibling::div"
                        return driver.find_element(By.XPATH, xpath).get_attribute("innerText").strip()
                    except:
                        return ""


                # Lấy dữ liệu chi tiết
                ma_TT = get_text_by_label("Mã thủ tục:")
                so_QD = get_text_by_label("Số quyết định:")
                ten_TT = get_text_by_label("Tên thủ tục:")
                cap_TH = get_text_by_label("Cấp thực hiện:")
                loai_TT = get_text_by_label("Loại thủ tục:")
                linh_vuc = get_text_by_label("Lĩnh vực:")
                trinh_tu_TH = get_text_by_label("Trình tự thực hiện:")
                doi_tuong_TH = get_text_by_label("Đối tượng thực hiện:")
                co_quan_TH = get_text_by_label("Cơ quan thực hiện:")
                co_quan_co_TQ = get_text_by_label("Cơ quan có thẩm quyền:")
                dia_chi_tiep_nhan = get_text_by_label("Địa chỉ tiếp nhận hồ sơ:")
                co_quan_duoc_UQ = get_text_by_label("Cơ quan được ủy quyền:")
                co_quan_PH = get_text_by_label("Cơ quan phối hợp:")
                ket_qua_TH = get_text_by_label("Kết quả thực hiện:")
                yeu_cau_dieu_kien = get_text_by_label("Yêu cầu, điều kiện thực hiện:")
                mo_ta = get_text_by_label("Mô tả:")

                writer.writerow([ma_TT, so_QD, ten_TT, cap_TH, loai_TT, linh_vuc, trinh_tu_TH, doi_tuong_TH,
                                 co_quan_TH, co_quan_co_TQ, dia_chi_tiep_nhan, co_quan_duoc_UQ,
                                 co_quan_PH, ket_qua_TH, yeu_cau_dieu_kien, mo_ta])

                driver.close()  # Đóng tab chi tiết
                driver.switch_to.window(main_tab)  # Quay lại tab chính
                time.sleep(1)

            except Exception as e:
                print(f"Lỗi dòng {i + 1} trang {page}: {e}")
                continue

        # Chuyển trang nếu còn trang tiếp theo
        try:
            next_btn = driver.find_element(By.XPATH, "/html/body/cache/div/div/div/div/div[1]/div[7]/div[2]/ul/li[8]")
            if "disabled" in next_btn.get_attribute("class"):
                break  # Nếu đã tới trang cuối
            next_btn.click()
            time.sleep(2)
        except:
            break

driver.quit()
