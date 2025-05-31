import os
import pandas as pd
import glob

def juntar_csvs(diretorio_entrada, arquivo_saida):
    """
    Função para juntar múltiplos arquivos CSV em um único arquivo.
    
    Parâmetros:
    diretorio_entrada: Diretório onde estão os arquivos CSV
    arquivo_saida: Nome do arquivo CSV de saída
    """
    # Lista para armazenar os dataframes
    dfs = []
    
    # Obtém a lista de arquivos CSV no diretório
    arquivos_csv = glob.glob(os.path.join(diretorio_entrada, "*.csv"))
    
    # Ordena os arquivos para garantir uma ordem consistente
    arquivos_csv.sort()
    
    # Adiciona uma coluna com o nome do arquivo para identificação
    for i, arquivo in enumerate(arquivos_csv):
        # Extrai o nome do arquivo sem o caminho completo
        nome_arquivo = os.path.basename(arquivo)
        
        try:
            # Tenta ler o arquivo CSV com encoding 'latin1', pulando a primeira linha (comentário)
            # e usando ponto e vírgula como separador
            df = pd.read_csv(arquivo, sep=';', skiprows=1, encoding='latin1')
            
            # Adiciona uma coluna com o nome do arquivo para identificação
            df['arquivo_origem'] = nome_arquivo
            
            # Adiciona o dataframe à lista
            dfs.append(df)
            print(f"Arquivo processado com sucesso: {nome_arquivo}")
        except Exception as e:
            print(f"Erro ao processar o arquivo {nome_arquivo}: {str(e)}")
    
    if dfs:
        # Concatena todos os dataframes
        df_final = pd.concat(dfs, ignore_index=True)
        
        # Salva o dataframe final em um arquivo CSV
        df_final.to_csv(arquivo_saida, sep=';', index=False, encoding='latin1')
        
        print(f"\nArquivos unidos com sucesso! Resultado salvo em: {arquivo_saida}")
        print(f"Total de arquivos processados: {len(arquivos_csv)}")
        print(f"Total de linhas no arquivo final: {len(df_final)}")
    else:
        print("Nenhum arquivo foi processado com sucesso.")

# Exemplo de uso
if __name__ == "__main__":
    diretorio_entrada = r"C:\Users\lucas\OneDrive\Área de Trabalho\Projetos_git\analise_banco\Dados - ressalto - aula do dia 3-3-2025"  # Seu diretório com os arquivos CSV
    arquivo_saida = r"C:\Users\lucas\OneDrive\Área de Trabalho\Projetos_git\analise_banco\dados_unidos.csv"  # Caminho para salvar o arquivo final
    
    juntar_csvs(diretorio_entrada, arquivo_saida)

