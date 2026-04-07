# import javalang
# import pandas as pd
# import os

# def extract_java_features(java_path):
#     """
#     Parses a Java file to extract structural AST nodes and semantic code tokens.
#     Uses the naming convention: NodeType_NodeName (e.g., MethodDeclaration_getRoute)
#     """
#     if not os.path.exists(java_path):
#         return None, None 

#     try:
#         with open(java_path, 'r', encoding='utf-8', errors='ignore') as f:
#             code = f.read()
        
#         # 1. AST Extraction (Structural Level)
#         tree = javalang.parse.parse(code)
        
#         # We capture the Node Type and the specific Name (if it exists) to enrich the sequence
#         ast_nodes = []
#         for _, node in tree:
#             if isinstance(node, (javalang.tree.Declaration, 
#                                  javalang.tree.MethodInvocation, 
#                                  javalang.tree.Statement)):
#                 node_type = type(node).__name__
#                 node_name = getattr(node, 'name', '') # Get name if available (e.g. method name)
#                 ast_nodes.append(f"{node_type}_{node_name}")
        
#         # 2. Token Extraction (Semantic Level)
#         # Filters out separators and operators to keep the vocabulary focused on identifiers
#         tokens = [t.value for t in javalang.tokenizer.tokenize(code) 
#                   if not isinstance(t, (javalang.tokenizer.Separator, javalang.tokenizer.Operator))]
        
#         if ast_nodes and tokens:
#             return " ".join(ast_nodes), " ".join(tokens)
#         return None, None

#     except Exception:
#         # Silently skip files that are severely corrupted or empty
#         return None, None

# def run_extraction(project_name, versions):
#     """
#     Standardized loop to process multiple versions of a project.
#     """
#     for v in versions:
#         # Directory Logic: data/project/project-v.csv
#         base_path = os.path.join("data", project_name)
#         csv_input = os.path.join(base_path, f"{project_name}-{v}.csv")
#         src_folder = os.path.join(base_path, f"src_{v}")
#         csv_output = os.path.join(base_path, f"{project_name}-{v}_enriched.csv")

#         if not os.path.exists(csv_input):
#             print(f"⚠️ Skipping {project_name} v{v}: {csv_input} not found.")
#             continue

#         print(f"--- 🛠️ Processing {project_name.upper()} Version {v} ---")
#         df = pd.read_csv(csv_input)
        
#         def process_row(filename):
#             # Maps 'org.apache.camel.Main' to 'org/apache/camel/Main.java'
#             path_suffix = filename.replace('.', '/') + ".java"
#             full_path = os.path.join(src_folder, path_suffix)
#             return extract_java_features(full_path)

#         # Extraction logic
#         print(f"Extracting features for {len(df)} files...")
#         features = df.iloc[:, 0].apply(lambda x: pd.Series(process_row(x)))
#         df[['ast_seq', 'code_tokens']] = features

#         # Data Cleaning: Keep only rows that were successfully parsed
#         df_cleaned = df.dropna(subset=['ast_seq', 'code_tokens'])
#         df_cleaned = df_cleaned[df_cleaned['ast_seq'] != ""]

#         print(f"✅ Success: {len(df_cleaned)}/{len(df)} files enriched.")
        
#         # Save to the project subfolder
#         df_cleaned.to_csv(csv_output, index=False)

# if __name__ == "__main__":
#     # To run Camel 1.4 -> 1.6
#     run_extraction("camel", ["1.4", "1.6"])
    
#     # To run Ant versions
#     # run_extraction("ant", ["1.3", "1.4", "1.5", "1.6", "1.7"])

import javalang
import pandas as pd
import os

def extract_java_features(java_path):
    """
    Parses a Java file to extract structural AST nodes and semantic code tokens.
    """
    try:
        with open(java_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        
        # 1. AST Extraction
        tree = javalang.parse.parse(code)
        ast_nodes = []
        for _, node in tree:
            if isinstance(node, (javalang.tree.Declaration, 
                                 javalang.tree.MethodInvocation, 
                                 javalang.tree.Statement)):
                node_type = type(node).__name__
                node_name = getattr(node, 'name', '')
                ast_nodes.append(f"{node_type}_{node_name}")
        
        # 2. Token Extraction
        tokens = [t.value for t in javalang.tokenizer.tokenize(code) 
                  if not isinstance(t, (javalang.tokenizer.Separator, javalang.tokenizer.Operator))]
        
        if ast_nodes and tokens:
            return " ".join(ast_nodes), " ".join(tokens)
    except Exception:
        pass
    return None, None

def find_file_recursively(root_folder, java_classname):
    """
    Searches the entire directory tree for the matching .java file.
    Also handles inner classes by splitting at '$'.
    """
    # Handle inner classes (org.apache.camel.Main$1 -> Main.java)
    base_file_name = java_classname.split('$')[0].split('.')[-1] + ".java"
    
    for root, dirs, files in os.walk(root_folder):
        if base_file_name in files:
            return os.path.join(root, base_file_name)
    return None

def run_extraction(project_name, versions):
    """
    Standardized loop with Deep Recursive Search.
    """
    for v in versions:
        base_path = os.path.join("data", project_name)
        csv_input = os.path.join(base_path, f"{project_name}-{v}.csv")
        src_folder = os.path.join(base_path, f"src_{v}")
        csv_output = os.path.join(base_path, f"{project_name}_{v}_enriched.csv")

        if not os.path.exists(csv_input):
            print(f"⚠️ Skipping {v}: {csv_input} not found.")
            continue

        print(f"--- 📂 Deep Processing {project_name.upper()} Version {v} ---")
        df = pd.read_csv(csv_input)
        
        all_ast = []
        all_tokens = []
        
        print(f"Scanning directory tree for {len(df)} files...")
        
        for index, row in df.iterrows():
            classname = row.iloc[0]
            # Perform the deep recursive search
            actual_path = find_file_recursively(src_folder, classname)
            
            ast, tokens = None, None
            if actual_path:
                ast, tokens = extract_java_features(actual_path)
            
            all_ast.append(ast)
            all_tokens.append(tokens)

        df['ast_seq'] = all_ast
        df['code_tokens'] = all_tokens

        # Clean: remove rows where source was not found or unparseable
        df_cleaned = df.dropna(subset=['ast_seq', 'code_tokens'])
        df_cleaned = df_cleaned[df_cleaned['ast_seq'] != ""]

        print(f"✅ Success: {len(df_cleaned)}/{len(df)} files enriched.")
        df_cleaned.to_csv(csv_output, index=False)

if __name__ == "__main__":
    run_extraction("synapse", [ "1.1", "1.2"])
    