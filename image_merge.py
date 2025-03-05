from PIL import Image
from pylatex import Document, Table, NoEscape
from pdf2image import convert_from_path
import os

def merge_images(image_paths, output_path, grid_size=(2, 4), image_size=(100, 100), spacing=10, background_color=(255, 255, 255)):
    """
    指定された複数の画像をグリッド状に結合する。
    
    :param image_paths: 画像ファイルのパスリスト
    :param output_path: 出力ファイルのパス
    :param grid_size: (行数, 列数) のタプル
    :param image_size: 各画像のサイズ (幅, 高さ)
    :param spacing: 画像間の間隔（ピクセル単位）
    :param background_color: 背景色（デフォルトは白）
    """
    images = [Image.open(img).resize(image_size) for img in image_paths]
    
    rows, cols = grid_size
    total_width = cols * image_size[0] + (cols - 1) * spacing
    total_height = rows * image_size[1] + (rows - 1) * spacing
    
    new_img = Image.new('RGB', (total_width, total_height), background_color)
    
    for index, img in enumerate(images):
        row = index // cols
        col = index % cols
        x_offset = col * (image_size[0] + spacing)
        y_offset = row * (image_size[1] + spacing)
        new_img.paste(img, (x_offset, y_offset))
    
    new_img.save(output_path)
    print(f"Merged image saved to {output_path}")


def latex_table_to_image(latex_table, output_image='table_image.png', dpi=300):
    """
    LaTeX形式の表を画像として出力する。
    
    :param latex_table: LaTeXの表の文字列
    :param output_image: 出力する画像のファイル名
    :param dpi: 画像の解像度（デフォルト300）
    """
    temp_tex = 'temp_table.tex'
    temp_pdf = 'temp_table.pdf'
    
    # LaTeX文書を作成
    doc = Document()
    doc.append(NoEscape(latex_table))
    doc.generate_pdf(temp_pdf.replace('.pdf', ''), clean_tex=False, clean=False)
    
    # PDFを画像に変換
    images = convert_from_path(temp_pdf, dpi=dpi)
    images[0].save(output_image, 'PNG')
    
    # 一時ファイルを削除
    os.remove(temp_tex)
    os.remove(temp_pdf)
    os.remove(temp_pdf.replace('.pdf', '.log'))
    os.remove(temp_pdf.replace('.pdf', '.aux'))
    
    print(f"Table image saved as {output_image}")

    return None

# 使用例 merge_images
# img_dir = 'C:/Users/al191/Documents/lab/performance_estimation/estimation_activity_level/program/pointcloud_view_calb/img'  # 画像が保存されているディレクトリ
# image_files = [os.path.join(img_dir, fname) for fname in sorted(os.listdir(img_dir)) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]  # imgディレクトリ内の画像を自動取得
# output_file = os.path.join(img_dir, 'merged_image.jpg')  # 出力ファイルをimgディレクトリ内に保存
# merge_images(image_files, output_file, grid_size=(3, 3), image_size=(750, 550))

# 使用例 latex_table_to_image
latex_table = r"""
\begin{table}[b]
		\caption{Evaluation metrics of each Random Forest model (short-video data)}
		\centering
		\resizebox{\columnwidth}{!}{%
			\begin{tabular}{cllllll} \hline
				Feature                      						& Accuracy & FNR   & FPR   & Precision & Recall & F1      \\ \hline \hline
				Image-based 50th percentile                         & 0.571    & 0.578 & 0.329 & 0.464     & 0.442  & 0.453   \\
				Image-based 90th percentile                         & 0.544    & 0.653 & 0.324 & 0.633     & 0.367  & 0.378   \\
				Center-of-gravity-based 50th percentile             & 0.566    & 0.497 & 0.388 & 0.447     & 0.401  & 0.423   \\
				Center-of-gravity-based 90th percentile             & 0.533    & 0.585 & 0.388 & 0.401     & 0.327  & 0.362   \\
				All                          						& 0.612    & 0.578 & 0.260 & 0.464     & 0.435  & 0.449   \\ \hline
			\end{tabular}
		}
		\label{result_rf_ext_train}
\end{table}
"""

latex_table_to_image(latex_table, output_image='table_image.png')


