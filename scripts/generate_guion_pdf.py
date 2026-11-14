import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def build_pdf():
    pdf_path = "/Users/johan6smith/Documents/CUN/APRENDIZAJE AUTOMATICO CICLO 2/aca2-deep-learning/video/Guion_Sustentacion_3min_CardioPredict_AI_2.pdf"
    alt_pdf_path = "/Users/johan6smith/Documents/CUN/APRENDIZAJE AUTOMATICO CICLO 2/aca2-deep-learning/video/guion_video_sustentacion_3min.pdf"
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=4,
        alignment=0
    )
    
    sub_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2563eb'),
        spaceAfter=10
    )
    
    meta_style = ParagraphStyle(
        'MetaText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#475569')
    )
    
    section_title = ParagraphStyle(
        'SecTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#0f172a')
    )
    
    time_badge = ParagraphStyle(
        'TimeBadge',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#b91c1c')
    )
    
    visual_cue = ParagraphStyle(
        'VisualCue',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1e40af')
    )
    
    speech_text = ParagraphStyle(
        'SpeechText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14.5,
        textColor=colors.HexColor('#0f172a')
    )
    
    tip_text = ParagraphStyle(
        'TipText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#334155')
    )

    story = []
    
    # Header Banner
    header_data = [
        [
            Paragraph("<b>🫀 GUION DE SUSTENTACIÓN EN VIDEO (3:00 MIN)</b>", title_style),
            Paragraph("<b>CRONÓMETRO:</b><br/><font color='#dc2626'><b>MÁX. 180 SEG</b></font>", time_badge)
        ],
        [
            Paragraph("<b>Proyecto:</b> CardioPredict AI 2.0 (Deep Learning & Triaje Clínico SQL)", sub_style),
            Paragraph("<b>Ponente:</b> Johan Smith Valderrama", meta_style)
        ]
    ]
    t_header = Table(header_data, colWidths=[4.2*inch, 2.5*inch])
    t_header.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_header)
    
    # Instructions Box
    instructions = [
        [
            Paragraph("""
            <b>📱 INSTRUCCIÓN PARA SUSTENTAR:</b> Abre este PDF en tu <b>teléfono celular</b> para leerlo fluidamente. 
            En la <b>pantalla de tu computador</b>, proyecta el PDF del póster (8 láminas) en pantalla completa. 
            Habla a un ritmo seguro (~135 palabras por minuto) y cambia de lámina cuando lo indique el cronómetro.
            """, tip_text)
        ]
    ]
    t_inst = Table(instructions, colWidths=[6.7*inch])
    t_inst.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_inst)
    story.append(Spacer(1, 8))
    
    # Slides / Cards
    slides = [
        {
            "num": "LÁMINA 1",
            "time": "⏱️ 0:00 - 0:20 (20 seg)",
            "title": "Portada Institucional y Presentación del Equipo",
            "visual": "🖥️ <b>EN PANTALLA:</b> Mostrar Lámina 1 (CardioPredict AI 2.0, CUN, Dr. Hamilton Rivera, Grupo 12).",
            "speech": "\"Un cordial saludo, profesor Hamilton y compañeros. En representación del <b>Grupo 12</b>, conformado por Marcela Avila, Jaime Cabrera, Cristhian Garzón y quien les habla, Johan Smith Valderrama, presento <b>CardioPredict AI 2.0</b>: un sistema inteligente de triaje clínico y detección de cardiopatía isquémica que integra <b>Deep Learning en PyTorch</b> y <b>gestión relacional en SQL</b> sobre datos multicéntricos.\""
        },
        {
            "num": "LÁMINA 2",
            "time": "⏱️ 0:20 - 0:45 (25 seg)",
            "title": "El Falso Negativo como Enemigo Clínico: El Problema",
            "visual": "🖥️ <b>EN PANTALLA:</b> Mostrar Lámina 2 (Infografía de 3 columnas: Problema -> Factores -> Desafío).",
            "speech": "\"Las enfermedades cardiovasculares son la primera causa de muerte mundial. En urgencias, hasta un 30% de pacientes presentan síntomas atípicos o silentes, y aquí <b>un falso negativo es fatal</b>: enviar a casa a un paciente infartado cuesta vidas. Por eso, nuestro objetivo no fue solo predecir, sino construir un ecosistema con <b>Sensibilidad superior al 93% y cero falsos negativos tolerables</b>, capturando factores complejos como la depresión del segmento ST y crisis hipertensivas.\""
        },
        {
            "num": "LÁMINA 3",
            "time": "⏱️ 0:45 - 1:10 (25 seg)",
            "title": "Unificación Multicéntrica y Gestión Relacional en SQLite",
            "visual": "🖥️ <b>EN PANTALLA:</b> Mostrar Lámina 3 (Diagrama de los 4 hospitales hacia cardiopredict.db).",
            "speech": "\"Cumpliendo el Paso 1 de la rúbrica, abandonamos los archivos planos y unificamos <b>918 pacientes de 4 centros hospitalarios internacionales</b>: Cleveland, Zurich, Hungría y Long Beach, dentro de una base de datos relacional SQLite llamada <b>cardiopredict.db</b>. Mediante tablas indexadas, vistas clínicas y un pipeline con StandardScaler y One-Hot Encoding, transformamos estas variables en tensores normalizados de 15 dimensiones listos para aprendizaje profundo.\""
        },
        {
            "num": "LÁMINA 4",
            "time": "⏱️ 1:10 - 1:35 (25 seg)",
            "title": "Diagrama Integral de Solución: El Ecosistema",
            "visual": "🖥️ <b>EN PANTALLA:</b> Mostrar Lámina 4 (Flujo en 3 fases: Ingesta SQL -> Núcleo PyTorch -> Triaje Web).",
            "speech": "\"Nuestra solución integra un flujo continuo en tres fases: Primero, la <b>Ingesta y Curación SQL</b> con imputación clínica. Segundo, el <b>Núcleo de Deep Learning en PyTorch</b>, donde entrenamos dos arquitecturas: una Red Convolucional 1D para bioseñales y nuestro modelo campeón, una Red Neuronal Densa Profunda. Y tercero, el <b>Despliegue Web en Streamlit</b> bajo un servidor Linux en la nube en iacun.lat.\""
        },
        {
            "num": "LÁMINA 5",
            "time": "⏱️ 1:35 - 2:00 (25 seg)",
            "title": "Topología de Red Profunda y Penalización Matemática",
            "visual": "🖥️ <b>EN PANTALLA:</b> Mostrar Lámina 5 (Red 15-64-32-16 y cuadro rojo de penalización asimétrica).",
            "speech": "\"La arquitectura de <b>CardioPredictDeepMLP</b> cuenta con tres capas ocultas de 64, 32 y 16 neuronas con activación ReLU y regularización Dropout para evitar sobreajuste. Pero el verdadero diferenciador clínico es su matemática: usamos la función BCEWithLogitsLoss con una <b>penalización asimétrica de factor 1.2</b> y un <b>umbral calibrado en 0.45</b>. Esto castiga severamente el error de clasificar a un paciente enfermo como sano, priorizando la sensibilidad médica.\""
        },
        {
            "num": "LÁMINA 6",
            "time": "⏱️ 2:00 - 2:20 (20 seg)",
            "title": "Salto Cuantitativo: Machine Learning vs. Deep Learning",
            "visual": "🖥️ <b>EN PANTALLA:</b> Mostrar Lámina 6 (Tabla comparativa: Recall 85% -> 93.14%, AUC 90.32%).",
            "speech": "\"Los resultados demuestran la superioridad del Deep Learning frente al corte anterior: Elevamos la <b>Sensibilidad del 85.29% al 93.14%</b>, alcanzando un <b>Área ROC del 90.32%</b>. En la práctica clínica real, detectamos a <b>95 de cada 102 pacientes con infarto</b>, reduciendo los falsos negativos de 15 a solo 7 casos; es decir, <b>más del 53% de reducción en errores letales</b> frente al ACA 1.\""
        },
        {
            "num": "LÁMINA 7",
            "time": "⏱️ 2:20 - 2:40 (20 seg)",
            "title": "Inferencias en Tiempo Real: El Semáforo Clínico de Triaje",
            "visual": "🖥️ <b>EN PANTALLA:</b> Mostrar Lámina 7 (Dashboard Streamlit y semáforo tricolor Rojo/Amarillo/Verde).",
            "speech": "\"En producción, nuestra interfaz implementa un <b>Semáforo Clínico de 3 Niveles</b>: <b>Rojo (>= 45%):</b> Prioridad I de alto riesgo para cateterismo urgente. <b>Ámbar (20% - 44%):</b> Una zona gris de observación preventiva para casos ambiguos con dolor atípico que no deben darse de alta. Y <b>Verde (&lt; 20%):</b> Bajo riesgo ambulatorio. Además, incluye presets de carga en un clic y auditoría SQL en vivo.\""
        },
        {
            "num": "LÁMINA 8",
            "time": "⏱️ 2:40 - 3:00 (20 seg)",
            "title": "Conclusiones, Ética Algorítmica y Despliegue en Producción",
            "visual": "🖥️ <b>EN PANTALLA:</b> Mostrar Lámina 8 (Conclusiones, código QR y enlaces oficiales).",
            "speech": "\"En conclusión, CardioPredict AI 2.0 trasciende el experimento de laboratorio: combina ética algorítmica, rigor en bases de datos relacionales y Deep Learning en un servidor operativo 24/7. Los invitamos a escanear el código QR o acceder a <b>https://iacun.lat/aprendizajeautomatico/aca2/</b> y consultar nuestro repositorio público en GitHub. ¡Muchas gracias por su atención!\""
        }
    ]
    
    for s in slides:
        card_content = [
            [
                Paragraph(f"<b>{s['num']}: {s['title']}</b>", section_title),
                Paragraph(s['time'], time_badge)
            ],
            [
                Paragraph(s['visual'], visual_cue),
                Paragraph("", meta_style)
            ],
            [
                Paragraph(s['speech'], speech_text),
                Paragraph("", meta_style)
            ]
        ]
        
        t_card = Table(card_content, colWidths=[5.4*inch, 1.3*inch])
        t_card.setStyle(TableStyle([
            ('SPAN', (0,1), (1,1)),
            ('SPAN', (0,2), (1,2)),
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#ffffff')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
            ('LINEBELOW', (0,0), (1,0), 1, colors.HexColor('#e2e8f0')),
            ('LINEBELOW', (0,1), (1,1), 0.5, colors.HexColor('#f1f5f9')),
            ('BACKGROUND', (0,0), (1,0), colors.HexColor('#f8fafc')),
            ('BACKGROUND', (0,1), (1,1), colors.HexColor('#eff6ff')),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        
        story.append(t_card)
        story.append(Spacer(1, 6))
        
    doc.build(story)
    
    import shutil
    shutil.copy(pdf_path, alt_pdf_path)
    print(f"Generated successfully: {pdf_path}")

if __name__ == "__main__":
    build_pdf()
