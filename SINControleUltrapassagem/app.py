import streamlit as st
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt


# Função para buscar os dados com filtros
@st.cache_data
def fetch_data():
    df = pd.read_csv('data/regras.csv')
    return df


# Função para calcular o intervalo de tempo mínimo
def calcular_tempo_minimo(distancia, velocidade):
    # Convertendo a velocidade de km/h para m/s
    velocidade_m_s = velocidade / 3.6
    if velocidade_m_s == 0:
        return float('inf')  # Evitar divisão por zero
    return distancia / velocidade_m_s

# Definindo as variáveis fuzzy
@st.cache_resource
def define_variables():
    speed = ctrl.Antecedent(np.arange(0, 101, 0.5), 'Velocidade')
    front_distance = ctrl.Antecedent(np.arange(0, 101, 0.5), 'Distancia')
    t_pista = ctrl.Antecedent(np.arange(0, 101, 0.5), 'T(pista)')

    # Definindo as funções de pertinência para as variáveis
    speed['Muito baixa'] = fuzz.trimf(speed.universe, [0, 0, 20])
    speed['Baixa'] = fuzz.trimf(speed.universe, [10, 25, 40])
    speed['Média'] = fuzz.trimf(speed.universe, [30, 50, 70])
    speed['Alta'] = fuzz.trimf(speed.universe, [60, 75, 90])
    speed['Muito alta'] = fuzz.trimf(speed.universe, [80, 100, 100])

    front_distance['Muito pequena'] = fuzz.trimf(front_distance.universe, [0, 0, 20])
    front_distance['Pequena'] = fuzz.trimf(front_distance.universe, [10, 25, 40])
    front_distance['Média'] = fuzz.trimf(front_distance.universe, [30, 50, 70])
    front_distance['Grande'] = fuzz.trimf(front_distance.universe, [60, 75, 90])
    front_distance['Muito grande'] = fuzz.trimf(front_distance.universe, [80, 100, 100])

    t_pista['Livre + Boa + Nenhum obstáculo'] = fuzz.trimf(t_pista.universe, [0, 0, 20])
    t_pista['Livre + Média + Obstáculo menor'] = fuzz.trimf(t_pista.universe, [10, 25, 40])
    t_pista['Livre + Ruim + Obstáculo médio'] = fuzz.trimf(t_pista.universe, [30, 50, 70])
    t_pista['Livre + Boa + Obstáculo maior'] = fuzz.trimf(t_pista.universe, [60, 75, 90])
    t_pista['Obstruída'] = fuzz.trimf(t_pista.universe, [80, 100, 100])

    # Definindo a variável de saída
    output = ctrl.Consequent(np.arange(0, 101, 0.5), 'Resultado')
    output['Colisão'] = fuzz.trimf(output.universe, [0, 0, 33])
    output['Ultrapassagem arriscada'] = fuzz.trimf(output.universe, [20, 50, 80])
    output['Ultrapassagem segura'] = fuzz.trimf(output.universe, [67, 100, 100])

    return speed, front_distance, t_pista, output

def get_membership_value(value, antecedent):
    max_membership = None
    max_term = None
    for term_name, term_mf in antecedent.terms.items():
        membership_value = fuzz.interp_membership(antecedent.universe, term_mf.mf, value)
        if max_membership is None or membership_value > max_membership:
            max_membership = membership_value
            max_term = term_name
    return max_term

def create_rules(speed, front_distance, t_pista, output, tempo_minimo):
    rules = []

    # Regras baseadas no tempo mínimo
    if tempo_minimo <= 1.125:
        rules.append(ctrl.Rule(speed["Muito baixa"] | speed["Baixa"] | speed["Média"] | speed["Alta"] | speed["Muito alta"], output["Colisão"]))

    rules.append(ctrl.Rule(speed['Alta'] & (front_distance['Média'] | front_distance['Pequena']) & 
                        (t_pista['Livre + Boa + Nenhum obstáculo']), output['Ultrapassagem segura']))

    rules.append(ctrl.Rule(speed['Alta'] & (front_distance['Muito pequena'] | front_distance['Pequena']) & 
                        (t_pista['Livre + Boa + Nenhum obstáculo']), output['Ultrapassagem segura']))

    rules.append(ctrl.Rule(speed['Média'] & (front_distance['Muito pequena'] | front_distance['Pequena']) & 
                        (t_pista['Obstruída']), output['Colisão']))

    rules.append(ctrl.Rule(speed['Muito alta'] & (front_distance['Média'] | front_distance['Grande']) & 
                        (t_pista['Livre + Média + Obstáculo menor']), output['Ultrapassagem arriscada']))

    rules.append(ctrl.Rule(speed['Baixa'] & (front_distance['Muito pequena'] | front_distance['Pequena'] | front_distance['Média']) & 
                        (t_pista['Obstruída']), output['Colisão']))

    rules.append(ctrl.Rule(speed['Muito baixa'] & (front_distance['Grande'] | front_distance['Muito grande']) & 
                        (t_pista['Livre + Ruim + Obstáculo médio']), output['Ultrapassagem arriscada']))

    rules.append(ctrl.Rule(speed['Alta'] & (front_distance['Grande'] | front_distance['Muito grande']) & 
                        (t_pista['Obstruída']), output['Colisão']))

    rules.append(ctrl.Rule(speed['Média'] & (front_distance['Muito pequena'] | front_distance['Pequena']) & 
                        (t_pista['Livre + Ruim + Obstáculo médio']), output['Colisão']))

    rules.append(ctrl.Rule(speed['Baixa'] & (front_distance['Média'] | front_distance['Grande']) & 
                        (t_pista['Livre + Boa + Nenhum obstáculo']), output['Ultrapassagem segura']))

    rules.append(ctrl.Rule(speed['Muito alta'] & (front_distance['Pequena'] | front_distance['Muito pequena']) & 
                        (t_pista['Obstruída']), output['Colisão']))

    rules.append(ctrl.Rule(speed['Alta'] & (front_distance['Grande'] | front_distance['Média']) & 
                        (t_pista['Livre + Média + Obstáculo menor']), output['Ultrapassagem arriscada']))

    rules.append(ctrl.Rule(speed['Média'] & (front_distance['Muito pequena'] | front_distance['Pequena']) & 
                        (t_pista['Obstruída']), output['Colisão']))

    rules.append(ctrl.Rule(speed['Baixa'] & (front_distance['Média'] | front_distance['Grande']) & 
                        (t_pista['Obstruída']), output['Colisão']))

    rules.append(ctrl.Rule(speed['Muito baixa'] & (front_distance['Grande'] | front_distance['Muito grande']) & 
                        (t_pista['Livre + Ruim + Obstáculo médio']), output['Ultrapassagem arriscada']))




    return rules

def main():
    st.set_page_config(page_title="Sistema de Inferência Nebulosa", layout="wide", page_icon=":car:")

    st.sidebar.title("Variáveis")
    
    speed, front_distance, t_pista, output = define_variables()

    # Sidebar sliders para configuração das variáveis
    speed_input = st.sidebar.slider("Velocidade", 0.0, 100.0, 50.0, 0.5)
    front_distance_input = st.sidebar.slider("Distancia", 0.0, 100.0, 50.0, 0.5)
    t_pista_input = st.sidebar.slider("T(pista)", 0.0, 100.0, 50.0, 0.5)

    submit_button = st.sidebar.button(label='Calcular Inferência')

    if submit_button:

        # Calculando o intervalo de tempo mínimo
        tempo_minimo = calcular_tempo_minimo(front_distance_input, speed_input)
        rules = create_rules(speed, front_distance, t_pista, output, tempo_minimo)
        # Exibindo o tempo mínimo no sidebar
        
        st.sidebar.write(f":clock3: `{tempo_minimo:.2f}` segundos")
        t_pista_label = get_membership_value(t_pista_input, t_pista)
        st.sidebar.write(f"	:hole: `{t_pista_label}`")
        speed_label = get_membership_value(speed_input, speed)
        
        st.sidebar.write(f":racing_car: `{speed_label}`")
        front_distance_label = get_membership_value(front_distance_input, front_distance)
        st.sidebar.write(f":triangular_ruler: `{front_distance_label}` metros")
        
        
        
        # Construindo o sistema de controle
        tipping_ctrl = ctrl.ControlSystem(rules)
        tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

        # Configurando os inputs
        tipping.input['Velocidade'] = speed_input
        tipping.input['Distancia'] = front_distance_input
        tipping.input['T(pista)'] = t_pista_input
        
        # Computando a inferência
        try:
            tipping.compute()
            resultado = tipping.output['Resultado']
            resultado_texto = f"Resultado da Inferência: {resultado:.2f}"
            if resultado < 33:
                resultado_texto = f"Colisão ({resultado:.2f})"
                st.sidebar.error(f"{resultado_texto}")
            elif resultado < 67:
                resultado_texto = f"Ultrapassagem arriscada ({resultado:.2f})"
                st.sidebar.warning(f"{resultado_texto}")
            else:
                resultado_texto = f"Ultrapassagem segura ({resultado:.2f})"
                st.sidebar.success(f"{resultado_texto}")
        except ValueError as e:
            resultado_texto = f"Erro de cálculo!"
            st.sidebar.warning(f"{resultado_texto}")
        
        # Layout da página com duas colunas
        col1, col2 = st.columns(2)
        
        # Gráficos das variáveis na primeira coluna
        with col1:
            speed.view(sim=tipping)
            st.pyplot(plt)

            front_distance.view(sim=tipping)
            st.pyplot(plt)
            
        # Gráfico do resultado da inferência na segunda coluna
        with col2:

            t_pista.view(sim=tipping)
            st.pyplot(plt)

            output.view(sim=tipping)
            st.pyplot(plt)

        

if __name__ == "__main__":
    main()
