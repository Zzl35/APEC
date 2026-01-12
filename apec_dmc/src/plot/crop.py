
import fitz  # PyMuPDF

def crop_pdf(input_path, output_path, crop_box):
    pdf_document = fitz.open(input_path)
    for page in pdf_document:
        print(page.rect)
        page.set_cropbox(crop_box)
    pdf_document.save(output_path)
    pdf_document.close()

# 裁剪 PDF
input_pdf = '/home/ubuntu/duxinghao/APEC/apec_dmc/src/exp_local/plot_output0129/reusability.pdf'
output_pdf = '/home/ubuntu/duxinghao/APEC/apec_dmc/src/exp_local/plot_output0129/reusability1.pdf'
crop_rectangle = (0, 0, 2304, 1110) # 左上右下, density: (60, 120, 1100, 1220), reuse:(0, 0, 2304, 1110)
crop_pdf(input_pdf, output_pdf, crop_rectangle)
