import streamlit as st
import torch
import yaml
import os
import sys
import pandas as pd
from PIL import Image
from datetime import datetime
from torchvision import transforms
import json

# --- GESTION DES CHEMINS D'IMPORT ---
# Ajoute le dossier racine au path pour pouvoir importer 'models' et 'explain'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.cnn import DocumentClassifierCNN
    from explain.gradcam import GradCAM, overlay_heatmap
except ImportError as e:
    st.error(f"Erreur d'importation : {e}")
    st.stop()

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Digitup - Classification de Documents",
    page_icon="📄",
    layout="wide"
)

# --- CONSTANTES ---
MODEL_PATH = "outputs/best_model.pth"
CLASSES_PATH = "outputs/classes.yaml"
FEEDBACK_DIR = "feedback"


# --- FONCTIONS ---

@st.cache_resource
def load_system():
    """Charge le modèle et les classes une seule fois (Cache)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Charger les noms des classes
    if not os.path.exists(CLASSES_PATH):
        return None, None, None, "Fichier classes.yaml introuvable. Avez-vous entraîné le modèle ?"

    with open(CLASSES_PATH, "r") as f:
        classes = yaml.safe_load(f)

    # 2. Charger le modèle
    if not os.path.exists(MODEL_PATH):
        return None, None, None, f"Modèle introuvable à : {MODEL_PATH}"

    try:
        model = DocumentClassifierCNN(num_classes=len(classes), freeze_backbone=False)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        return None, None, None, f"Erreur chargement modèle : {e}"

    return model, classes, device, None


def save_feedback(image, pred_label, correct_label):
    """Sauvegarde l'image et la correction dans le dossier feedback."""
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Sauvegarder l'image
    img_filename = f"feedback_{timestamp}_{correct_label}.png"
    img_path = os.path.join(FEEDBACK_DIR, img_filename)
    image.save(img_path)

    # 2. Logger dans le CSV
    csv_path = os.path.join(FEEDBACK_DIR, "corrections_log.csv")
    log_entry = {
        "timestamp": timestamp,
        "image": img_filename,
        "prediction_ia": pred_label,
        "correction_user": correct_label
    }

    df = pd.DataFrame([log_entry])

    # Ajout sans en-tête si le fichier existe déjà
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)


# --- INTERFACE UTILISATEUR ---

st.title("🤖 Digitup Intelligent Classifier")
st.markdown("Système de tri automatique de documents administratifs avec explicabilité.")

# Chargement des ressources
model, classes, device, error_msg = load_system()

if error_msg:
    st.error(f"❌ Erreur critique : {error_msg}")
    st.stop()

# Sidebar pour upload
with st.sidebar:
    st.header("📥 Entrée")
    uploaded_file = st.file_uploader("Glissez votre document ici (Image)", type=["jpg", "png", "jpeg", "tif"])
    st.info(f"Modèle chargé sur : **{device}**")
    st.info(f"Classes gérées : {len(classes)}")

# Logique principale
if uploaded_file is not None:
    # 1. Affichage Image
    col_left, col_right = st.columns([1, 1])

    image = Image.open(uploaded_file).convert("RGB")

    with col_left:
        st.subheader("📄 Document Original")
        st.image(image, use_column_width=True, caption="Image brute")

    # 2. Prétraitement & Prédiction
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probabilities, 0)

    pred_label = classes[pred_idx.item()]

    # 3. Affichage Résultats
    with col_right:
        st.subheader("📊 Analyse IA")

        # Métrique principale
        color = "green" if conf.item() > 0.8 else "orange"
        st.markdown(f"### Classe : :{color}[**{pred_label}**]")
        st.metric("Indice de Confiance", f"{conf.item():.2%}")

        # Graphique
        probs_np = probabilities.cpu().numpy()
        chart_data = pd.DataFrame({"Classe": classes, "Confiance": probs_np})
        chart_data = chart_data.sort_values(by="Confiance", ascending=False).head(5)

        st.bar_chart(chart_data.set_index("Classe"))
    report_data = {
        "filename": uploaded_file.name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": pred_label,
        "confidence_score": float(conf.item()),
        "top_3_classes": [
            {"class": classes[i], "score": float(probs_np[i])}
            for i in probs_np.argsort()[-3:][::-1]
        ]
    }

    report_json = json.dumps(report_data, indent=4)

    st.download_button(
        label="📥 Télécharger le Rapport d'Analyse (JSON)",
        data=report_json,
        file_name=f"rapport_{uploaded_file.name}.json",
        mime="application/json"
    )

    # 4. Explicabilité (Grad-CAM)
    st.divider()
    st.subheader("🧠 Explicabilité (Grad-CAM)")
    st.markdown("Les zones en **rouge** indiquent ce que l'IA a regardé pour prendre sa décision.")

    col_cam1, col_cam2 = st.columns([1, 3])

    with col_cam1:
        st.write("Génération de la heatmap...")
        target_layer = model.get_last_conv_layer()
        grad_cam = GradCAM(model, target_layer)
        heatmap, _ = grad_cam.generate_heatmap(input_tensor, class_idx=pred_idx.item())

        cam_image = overlay_heatmap(heatmap, image, alpha=0.5)
        st.image(cam_image, caption=f"Zones d'intérêt pour '{pred_label}'", use_column_width=True)

    with col_cam2:
        st.info("""
        **Comment lire cette image ?**
        - L'IA cherche souvent des logos, des en-têtes spécifiques ou la structure des paragraphes.
        - Si la zone rouge est sur le logo d'une marque, l'IA a reconnu la 'Publicité'.
        - Si la zone rouge couvre tout le texte dense, l'IA a reconnu un 'Rapport' ou une 'Lettre'.
        """)

    # 5. Feedback Loop
    st.divider()
    st.subheader("✍️ Correction & Amélioration")

    with st.expander("L'IA s'est trompée ? Cliquez ici pour corriger."):
        st.write("Aidez-nous à réentraîner le modèle en indiquant la bonne réponse.")

        with st.form("feedback_form"):
            correct_label = st.selectbox("Quelle est la vraie catégorie ?", classes, index=pred_idx.item())
            submit_btn = st.form_submit_button("Envoyer la correction")

            if submit_btn:
                if correct_label == pred_label:
                    st.warning("La correction est identique à la prédiction. Êtes-vous sûr ?")
                else:
                    save_feedback(image, pred_label, correct_label)
                    st.success(f"✅ Merci ! Feedback enregistré : L'image est un(e) **{correct_label}**.")
                    st.balloons()

else:
    st.info("👋 Bonjour ! Veuillez charger un document dans le menu de gauche pour commencer.")