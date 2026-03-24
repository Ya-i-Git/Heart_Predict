import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sklearn

# Устанавливаем вывод трансформеров в pandas (удобно для отладки)
sklearn.set_config(transform_output="pandas")

# Загрузка модели
@st.cache_resource
def load_model():
    return joblib.load("cat_pipeline.pkl")

pipeline = load_model()

# Ожидаемые колонки (можно взять из модели, но для надёжности зададим явно)
EXPECTED_COLUMNS = list(pipeline.feature_names_in_)

# Категориальные признаки и их допустимые значения (в формате, как они хранятся в модели)
CATEGORICAL_FEATURES = {
    "Sex": ["M", "F"],
    "ChestPainType": ["ATA", "NAP", "ASY", "TA"],
    "RestingECG": ["Normal", "ST", "LVH"],
    "ExerciseAngina": ["N", "Y"],
    "FastingBS": [0, 1],
    "ST_Slope": ["Up", "Flat", "Down"]
}

# Русские названия полей
FIELD_NAMES_RU = {
    "Age": "Возраст (лет)",
    "Sex": "Пол",
    "ChestPainType": "Тип боли в груди",
    "RestingBP": "Артериальное давление в покое (мм рт. ст.)",
    "Cholesterol": "Холестерин (мг/дл)",
    "FastingBS": "Глюкоза натощак > 120 мг/дл",
    "RestingECG": "Результаты ЭКГ в покое",
    "MaxHR": "Максимальная ЧСС (уд/мин)",
    "ExerciseAngina": "Стенокардия при нагрузке",
    "Oldpeak": "Депрессия ST",
    "ST_Slope": "Наклон сегмента ST"
}

# Перевод значений для отображения
CATEGORY_LABELS_RU = {
    "Sex": {"M": "Мужской", "F": "Женский"},
    "ChestPainType": {
        "ATA": "Атипичная стенокардия", "NAP": "Неангинозная боль",
        "ASY": "Асимптоматическая", "TA": "Типичная стенокардия"
    },
    "RestingECG": {"Normal": "Норма", "ST": "Аномалии ST-T", "LVH": "Гипертрофия левого желудочка"},
    "ExerciseAngina": {"N": "Нет", "Y": "Да"},
    "FastingBS": {0: "Нет", 1: "Да"},
    "ST_Slope": {"Up": "Восходящий", "Flat": "Плоский", "Down": "Нисходящий"}
}

# Боковая панель навигации
st.sidebar.title("Навигация")
page = st.sidebar.radio("Выберите страницу:", ["Указать симптомы", "Загрузить CSV"])

# Инициализация состояния
if "show_info" not in st.session_state:
    st.session_state.show_info = False

# Кнопка переключения
if st.sidebar.button("Информация о модели"):
    st.session_state.show_info = not st.session_state.show_info

# Отображение информации в зависимости от состояния
if st.session_state.show_info:
    st.sidebar.subheader("Ожидаемые колонки")
    st.sidebar.write(", ".join(EXPECTED_COLUMNS))
    st.sidebar.subheader("Типы колонок")
    numeric = [col for col in EXPECTED_COLUMNS if col not in CATEGORICAL_FEATURES]
    st.sidebar.write("Числовые:", ", ".join(numeric))
    st.sidebar.write("Категориальные:", ", ".join(CATEGORICAL_FEATURES.keys()))
    st.sidebar.subheader("Допустимые категориальные значения")
    for feat, vals in CATEGORICAL_FEATURES.items():
        ru_vals = [CATEGORY_LABELS_RU[feat].get(v, v) for v in vals]
        st.sidebar.write(f"{FIELD_NAMES_RU[feat]}: {', '.join(ru_vals)}")

    # Создаём пример CSV
    example_data = {}
    for col in EXPECTED_COLUMNS:
        if col in CATEGORICAL_FEATURES:
            example_data[col] = CATEGORICAL_FEATURES[col][0]
        else:
            if col == "Age":
                example_data[col] = 50
            elif col == "RestingBP":
                example_data[col] = 120
            elif col == "Cholesterol":
                example_data[col] = 200
            elif col == "MaxHR":
                example_data[col] = 150
            elif col == "Oldpeak":
                example_data[col] = 1.0
            else:
                example_data[col] = 0
    example_df = pd.DataFrame([example_data])
    csv_example = example_df.to_csv(index=False)
    st.sidebar.download_button(
        label="Скачать пример CSV",
        data=csv_example,
        file_name="example.csv",
        mime="text/csv"
    )

# ==================== Страница "Указать симптомы" ====================
if page == "Указать симптомы":
    st.markdown("<h1 style='text-align: center;'>🌳 Чеклист для пациента</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(FIELD_NAMES_RU["Age"], min_value=1, max_value=120, value=50)
        sex = st.selectbox(FIELD_NAMES_RU["Sex"], options=CATEGORICAL_FEATURES["Sex"],
                           format_func=lambda x: CATEGORY_LABELS_RU["Sex"][x])
        chest_pain = st.selectbox(FIELD_NAMES_RU["ChestPainType"], options=CATEGORICAL_FEATURES["ChestPainType"],
                                  format_func=lambda x: CATEGORY_LABELS_RU["ChestPainType"][x])
        resting_bp = st.number_input(FIELD_NAMES_RU["RestingBP"], min_value=0, max_value=300, value=120)
        cholesterol = st.number_input(FIELD_NAMES_RU["Cholesterol"], min_value=0, max_value=700, value=200)

    with col2:
        fasting_bs = st.selectbox(FIELD_NAMES_RU["FastingBS"], options=CATEGORICAL_FEATURES["FastingBS"],
                                  format_func=lambda x: CATEGORY_LABELS_RU["FastingBS"][x])
        resting_ecg = st.selectbox(FIELD_NAMES_RU["RestingECG"], options=CATEGORICAL_FEATURES["RestingECG"],
                                   format_func=lambda x: CATEGORY_LABELS_RU["RestingECG"][x])
        max_hr = st.number_input(FIELD_NAMES_RU["MaxHR"], min_value=50, max_value=250, value=150)
        exercise_angina = st.selectbox(FIELD_NAMES_RU["ExerciseAngina"], options=CATEGORICAL_FEATURES["ExerciseAngina"],
                                       format_func=lambda x: CATEGORY_LABELS_RU["ExerciseAngina"][x])
        oldpeak = st.number_input(FIELD_NAMES_RU["Oldpeak"], min_value=-5.0, max_value=10.0, value=0.0, step=0.1)

    st_slope = st.selectbox(FIELD_NAMES_RU["ST_Slope"], options=CATEGORICAL_FEATURES["ST_Slope"],
                            format_func=lambda x: CATEGORY_LABELS_RU["ST_Slope"][x])

    if st.button("Сделать прогноз пациенту"):
        input_data = pd.DataFrame([{
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "RestingECG": resting_ecg,
            "MaxHR": max_hr,
            "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope,
        }])

        # Убеждаемся, что порядок колонок совпадает
        input_data = input_data[EXPECTED_COLUMNS]

        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]

        st.markdown("---")
        if prediction == 1:
            st.error(f"⚠️ Высокий риск сердечного заболевания (вероятность: {probability:.2f})")
        else:
            st.success(f"✅ Низкий риск сердечного заболевания (вероятность: {probability:.2f})")

# ==================== Страница "Загрузить CSV" ====================
else:
    st.title("Сделать прогноз для пациента")
    st.write("Загрузите CSV-файл без колонки `HeartDisease`")
    st.info(f"Ожидаемые колонки: `{"`, `".join(EXPECTED_COLUMNS)}`")

    uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Загруженные данные")
        st.dataframe(df.head(10))

        # Валидация
        missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        extra = "HeartDisease" in df.columns

        if missing:
            st.error(f"Не хватает колонок: {missing}")
        elif extra:
            st.warning("Файл содержит колонку `HeartDisease` — она будет удалена перед предсказанием.")
            df = df.drop("HeartDisease", axis=1)

        if not missing:
            # Проверяем категориальные значения (необязательно, но полезно)
            for col, vals in CATEGORICAL_FEATURES.items():
                if col in df.columns:
                    invalid = ~df[col].isin(vals)
                    if invalid.any():
                        bad_vals = df.loc[invalid, col].unique()
                        st.warning(f"Колонка '{col}' содержит значения, не встречавшиеся при обучении: {bad_vals}. Это может вызвать ошибку.")

            # Приводим категориальные колонки к строковому типу (на всякий случай)
            for col in CATEGORICAL_FEATURES.keys():
                if col in df.columns:
                    df[col] = df[col].astype(str)

            # Предсказание
            predictions = pipeline.predict(df[EXPECTED_COLUMNS])
            probabilities = pipeline.predict_proba(df[EXPECTED_COLUMNS])[:, 1]

            df_result = df.copy()
            df_result["HeartDisease"] = predictions
            df_result["Probability"] = probabilities

            st.subheader("Результат предсказания")
            st.dataframe(df_result)

            # Статистика
            total = len(predictions)
            sick = int((predictions == 1).sum())
            healthy = total - sick

            st.subheader("Статистика")
            col1, col2, col3 = st.columns(3)
            col1.metric("Всего", total)
            col2.metric("Риск (1)", f"{sick} ({sick/total*100:.1f}%)")
            col3.metric("Нет риска (0)", f"{healthy} ({healthy/total*100:.1f}%)")

            fig_data = pd.DataFrame({"Класс": ["Нет риска", "Риск"], "Количество": [healthy, sick]})
            st.bar_chart(fig_data.set_index("Класс"))

            # Скачивание
            csv = df_result.to_csv(index=False)
            st.download_button(
                label="Скачать CSV с предсказаниями",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )