#!/usr/bin/env python3
"""
Script de teste para verificar dependências da Etapa II
"""

def testar_dependencias():
    """Testa se todas as dependências estão funcionando"""
    
    print("🔍 Testando dependências da Etapa II...")
    
    try:
        import streamlit as st
        print("✅ Streamlit:", st.__version__)
    except ImportError as e:
        print("❌ Streamlit:", e)
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas:", pd.__version__)
    except ImportError as e:
        print("❌ Pandas:", e)
        return False
    
    try:
        import numpy as np
        print("✅ NumPy:", np.__version__)
    except ImportError as e:
        print("❌ NumPy:", e)
        return False
    
    try:
        import plotly
        print("✅ Plotly:", plotly.__version__)
    except ImportError as e:
        print("❌ Plotly:", e)
        return False
    
    try:
        import scipy
        print("✅ SciPy:", scipy.__version__)
    except ImportError as e:
        print("❌ SciPy:", e)
        return False
    
    try:
        import statsmodels
        print("✅ Statsmodels:", statsmodels.__version__)
    except ImportError as e:
        print("❌ Statsmodels:", e)
        return False
    
    try:
        import sklearn
        print("✅ Scikit-learn:", sklearn.__version__)
    except ImportError as e:
        print("❌ Scikit-learn:", e)
        return False
    
    print("\n🎉 Todas as dependências estão funcionando!")
    return True

def testar_funcionalidades():
    """Testa funcionalidades específicas"""
    
    print("\n🔧 Testando funcionalidades...")
    
    try:
        import pandas as pd
        import numpy as np
        from scipy import stats
        from scipy.optimize import curve_fit
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import het_breuschpagan
        
        # Teste básico de regressão
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        
        # Regressão linear
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        print("✅ Regressão linear básica funcionando")
        
        # Statsmodels
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        print("✅ Statsmodels funcionando")
        
        # Curve fit
        def linear_func(x, a, b):
            return a * x + b
        
        params, _ = curve_fit(linear_func, x, y)
        print("✅ Curve fit funcionando")
        
        print("🎉 Todas as funcionalidades estão funcionando!")
        return True
        
    except Exception as e:
        print(f"❌ Erro nas funcionalidades: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("TESTE DE DEPENDÊNCIAS - ETAPA II")
    print("=" * 50)
    
    deps_ok = testar_dependencias()
    func_ok = testar_funcionalidades()
    
    print("\n" + "=" * 50)
    if deps_ok and func_ok:
        print("🎉 SUCESSO: Todas as dependências e funcionalidades estão OK!")
        print("✅ Você pode executar: streamlit run app4_modelagem.py")
    else:
        print("❌ PROBLEMA: Algumas dependências ou funcionalidades falharam!")
        print("💡 Execute: pip install -r requirements_etapa2.txt")
    print("=" * 50) 