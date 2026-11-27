import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def create_flowchart():
    fig = plt.figure(figsize=(16, 22), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 110)
    ax.axis('off')

    # Background canvas
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    # Color Palette
    c_blue = '#1E40AF'
    c_blue_bg = '#EFF6FF'
    c_teal = '#0F766E'
    c_teal_bg = '#F0FDFA'
    c_purple = '#6D28D9'
    c_purple_bg = '#FAF5FF'
    c_emerald = '#047857'
    c_emerald_bg = '#ECFDF5'
    c_amber = '#B45309'
    c_amber_bg = '#FFFBEB'
    c_border = '#CBD5E1'

    # Title
    ax.text(50, 106.5, "ARSITEKTUR & FLOWCHART FUSASHIELD", 
            ha='center', va='center', fontsize=22, fontweight='bold', color='#0F172A')
    ax.text(50, 104, "Sistem Mitigasi Presisi Layu Fusarium Berbasis IoT, MobileNetV2, Fuzzy Sugeno & Groq LLM", 
            ha='center', va='center', fontsize=12, fontstyle='italic', color='#475569')

    def draw_box(x, y, w, h, title, subtitle, bg_color, border_color, title_color='#0F172A', text_color='#334155'):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8,rounding_size=1.2",
                             linewidth=2, edgecolor=border_color, facecolor=bg_color, zorder=2)
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2, y + h*0.68, title, ha='center', va='center', fontsize=11, fontweight='bold', color=title_color, zorder=3)
            ax.text(x + w/2, y + h*0.32, subtitle, ha='center', va='center', fontsize=9.5, color=text_color, zorder=3)
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=11, fontweight='bold', color=title_color, zorder=3)

    def draw_group(x, y, w, h, label, color):
        group_box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.5",
                                   linewidth=1.5, linestyle='--', edgecolor=color, facecolor='#FFFFFF', alpha=0.9, zorder=1)
        ax.add_patch(group_box)
        ax.text(x + 2.5, y + h - 2.5, label, fontsize=11, fontweight='bold', color=color, zorder=2)

    def draw_arrow(x1, y1, x2, y2, label=None, color='#64748B'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=15), zorder=4)
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx, my + 1, label, ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=color,
                    bbox=dict(boxstyle='square,pad=0.2', facecolor='#FFFFFF', edgecolor='none'), zorder=5)

    # 1. TAHAP SENSING & AKUISISI
    draw_group(5, 84, 90, 16, "1. AKUISISI DATA SENSORIK & CITRA APLIKASI", c_blue)
    draw_box(10, 86, 36, 11, "Sensor Telemetri IoT (ESP32)", "• Suhu Lingkungan (°C)\n• Kelembapan Tanah (%)\n• Kelembapan Udara / RH (%)\n• Kirim real-time ke Firebase", c_blue_bg, c_blue)
    draw_box(54, 86, 36, 11, "Input Citra Aplikasi FusaShield", "• Pengambilan foto daun (1 hari sekali)\n• Difoto langsung lewat Aplikasi FusaShield\n• Upload otomatis ke Supabase Storage", c_teal_bg, c_teal)

    # 2. TAHAP PREPROCESSING & COMPUTER VISION
    draw_group(5, 62, 90, 18, "2. PREPROCESSING & INFERENSI CNN MOBILENETV2", c_teal)
    draw_box(10, 65, 36, 12, "Preprocessing Citra Digital", "• Validasi usia foto (<= 24 jam terakhir)\n• Koreksi orientasi EXIF kamera smartphone\n• Resize resolusi 224x224 RGB\n• Normalisasi piksel tensor", '#F8FAFC', c_border)
    draw_box(54, 65, 36, 12, "Model CNN MobileNetV2", "• Ekstraksi fitur visual transfer learning\n• Klasifikasi 6 kelas penyakit daun cabai\n• Output Probabilitas (P_sehat, P_klorosis)\n• Skor Gejala = (1-P_sehat)*60 + P_klorosis*40", c_teal_bg, c_teal)

    # Arrows 1 -> 2
    draw_arrow(28, 86, 28, 59, "Firebase RTDB")
    draw_arrow(72, 86, 28, 77, "Supabase Storage")
    draw_arrow(28, 65, 54, 71)

    # 3. TAHAP GERBANG VALIDASI SISTEM
    draw_box(20, 48, 60, 10, "GERBANG KELAYAKAN SISTEM (GATEKEEPER VALIDATION)", 
             "Syarat Mutlak: Telemetri IoT Lengkap + Foto Aplikasi <= 24 Jam Terakhir\n(Jika belum ada foto hari ini / sensor offline: Sistem STANDBY & Pompa OFF)", 
             c_amber_bg, c_amber, title_color=c_amber)

    draw_arrow(54, 65, 50, 58)
    draw_arrow(28, 59, 50, 58)

    # 4. TAHAP FUZZY SUGENO ORDE 0
    draw_group(5, 23, 90, 21, "4. SISTEM INFERENSI KEPUTUSAN FUZZY SUGENO (ORDE NOL)", c_purple)
    draw_box(10, 26, 26, 15, "Fuzzifikasi Trapesium", 
             "Variabel Input Linguistik:\n• RH Udara (Rendah, Sedang, Tinggi)\n• Tanah (Kering, Lembap, Basah)\n• Suhu (Dingin, Optimal, Panas)\n• Skor Citra (Ringan, Sedang, Berat)", 
             c_purple_bg, c_purple)

    draw_box(39, 26, 26, 15, "Evaluasi Rule Base", 
             "Inferensi MIN-MAX Firing:\n• r_kritis (Kondisi jamur aktif)\n• r_waspada (Potensi penularan)\n• r_preventif (Kondisi lembap)\n• r_aman (Tanaman sehat)", 
             c_purple_bg, c_purple)

    draw_box(68, 26, 22, 15, "Defuzzifikasi Sugeno", 
             "Weighted Average:\nZ = Σ(r_i * z_i) / Σ(r_i)\n\nKonstanta Takaran (ml):\n• Aman: 0 ml\n• Preventif: 20 ml\n• Waspada: 50 ml\n• Kritis: 100 ml", 
             c_purple_bg, c_purple)

    draw_arrow(50, 48, 23, 41, "Lolos Validasi (Data Riil)")
    draw_arrow(36, 33.5, 39, 33.5)
    draw_arrow(65, 33.5, 68, 33.5)

    # 5. TAHAP EXPLAINABLE AI & EKSEKUSI
    draw_group(5, 3, 90, 16, "5. EXPLAINABLE AI (XAI) & AKTUALISASI SISTEM", c_emerald)
    draw_box(10, 5, 26, 11, "Groq LLM (Llama 3.3 70B)", "• XAI Analisis agronomi ilmiah\n• Korelasi mikroklimat vs Fusarium\n• Rekomendasi fisik budidaya di lahan\n• Strict Guardrails (Dosis terkunci)", c_blue_bg, c_blue)
    draw_box(39, 5, 26, 11, "Aktuator Pompa Mikro ESP32", "• Konversi: Durasi = Dosis / 5.0 detik\n• Penyemprotan Trichoderma cair\n• Cooldown Protection (Kunci 12 Jam)\n• Feedback sinyal ACK durasi aktual", c_emerald_bg, c_emerald)
    draw_box(68, 5, 22, 11, "Web Dashboard (Vercel)", "• Telemetri live sensor Firebase\n• Display foto daun Supabase\n• Riwayat log mitigasi pompa\n• Narasi agronomi XAI real-time", c_teal_bg, c_teal)

    draw_arrow(79, 26, 23, 16)
    draw_arrow(79, 26, 52, 16)
    draw_arrow(52, 10.5, 68, 10.5, "Log Firebase")
    draw_arrow(23, 5, 68, 5, "Sync Dashboard")

    plt.tight_layout()
    output_path = r"c:\project\Fusashield\flowchart_fusashield.png"
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("Updated Flowchart PNG successfully saved to:", output_path)

if __name__ == "__main__":
    create_flowchart()
