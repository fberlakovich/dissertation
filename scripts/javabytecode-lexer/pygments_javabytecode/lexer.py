# file: pygments_javabytecode/lexer.py
from pygments.lexer import RegexLexer, bygroups, words, include
from pygments.token import (
    Text, Comment, Keyword, Name, String, Number, Operator, Punctuation, Generic
)

# JVM opcode mnemonics (hot path: keep this reasonably complete)
_JVM_OPS = [
    "nop","aconst_null","iconst_m1","iconst_0","iconst_1","iconst_2","iconst_3","iconst_4","iconst_5",
    "lconst_0","lconst_1","fconst_0","fconst_1","fconst_2","dconst_0","dconst_1","bipush","sipush","ldc",
    "ldc_w","ldc2_w","iload","lload","fload","dload","aload","iload_0","iload_1","iload_2","iload_3",
    "lload_0","lload_1","lload_2","lload_3","fload_0","fload_1","fload_2","fload_3","dload_0","dload_1",
    "dload_2","dload_3","aload_0","aload_1","aload_2","aload_3","iaload","laload","faload","daload","aaload",
    "baload","caload","saload","istore","lstore","fstore","dstore","astore","istore_0","istore_1","istore_2",
    "istore_3","lstore_0","lstore_1","lstore_2","lstore_3","fstore_0","fstore_1","fstore_2","fstore_3",
    "dstore_0","dstore_1","dstore_2","dstore_3","astore_0","astore_1","astore_2","astore_3","iastore",
    "lastore","fastore","dastore","aastore","bastore","castore","sastore","pop","pop2","dup","dup_x1","dup_x2",
    "dup2","dup2_x1","dup2_x2","swap","iadd","ladd","fadd","dadd","isub","lsub","fsub","dsub","imul","lmul",
    "fmul","dmul","idiv","ldiv","fdiv","ddiv","irem","lrem","frem","drem","ineg","lneg","fneg","dneg","ishl",
    "lshl","ishr","lshr","iushr","lushr","iand","land","ior","lor","ixor","lxor","iinc","i2l","i2f","i2d",
    "l2i","l2f","l2d","f2i","f2l","f2d","d2i","d2l","d2f","i2b","i2c","i2s","lcmp","fcmpl","fcmpg","dcmpl",
    "dcmpg","ifeq","ifne","iflt","ifge","ifgt","ifle","if_icmpeq","if_icmpne","if_icmplt","if_icmpge",
    "if_icmpgt","if_icmple","if_acmpeq","if_acmpne","goto","jsr","ret","tableswitch","lookupswitch","ireturn",
    "lreturn","freturn","dreturn","areturn","return","getstatic","putstatic","getfield","putfield",
    "invokevirtual","invokespecial","invokestatic","invokeinterface","invokedynamic","new","newarray",
    "anewarray","arraylength","athrow","checkcast","instanceof","monitorenter","monitorexit","wide",
    "multianewarray","ifnull","ifnonnull","goto_w","jsr_w"
]

# JVM access flags and section-ish keywords often seen in `javap -v`
_FLAGS = [
    "ACC_PUBLIC","ACC_PRIVATE","ACC_PROTECTED","ACC_STATIC","ACC_FINAL","ACC_SUPER","ACC_SYNCHRONIZED",
    "ACC_VOLATILE","ACC_TRANSIENT","ACC_NATIVE","ACC_INTERFACE","ACC_ABSTRACT","ACC_STRICT","ACC_SYNTHETIC",
    "ACC_ANNOTATION","ACC_ENUM","ACC_MODULE"
]

class JavaBytecodeLexer(RegexLexer):
    """
    Lexer for `javap` / JVM bytecode disassembly text (not raw .class).
    Examples: output of `javap -c` or `javap -v`.
    """
    name = "Java Bytecode (javap)"
    aliases = ["javabytecode", "java-bytecode", "javap"]
    filenames = ["*.javap", "*.jbc", "*.jasm"]  # loose; convenient defaults
    mimetypes = ["text/x-java-bytecode"]

    tokens = {
        "root": [
            # single-line comments (javap uses lots of // notes)
            (r"//.*?$", Comment.Single),

            # Headings & sections
            (r"^(Classfile|Compiled from|SourceFile|Constant pool:|Code:|"
             r"LineNumberTable:|LocalVariableTable:|BootstrapMethods:|"
             r"InnerClasses:|Exceptions:|Methods:|Fields:)\b.*$", Generic.Heading),

            # Method/class signatures lines
            (r"^(public|private|protected|static|final|abstract|\S+)\s+class\b.*$", Keyword.Declaration),
            (r"^(public|private|protected|static|final|native|abstract|"
             r"synchronized)\b.*\)$", Keyword.Declaration),

            # Flags
            (words(_FLAGS, suffix=r"\b"), Name.Attribute),

            # Left-margin bytecode line numbers:  "  12: "
            (r"^\s*\d+:(?=\s)", Number.Integer),

            # Labels like "L0", "L10:"
            (r"\bL\d+:?", Name.Label),

            # Constant-pool references: "#12" or "#12, #34"
            (r"#\d+", Name.Constant),

            # Opcodes/mnemonics
            (words(_JVM_OPS, suffix=r"\b"), Keyword),

            # Class-like internal names (java/lang/Object, [Ljava/lang/String;)
            (r"(?:\[+)?L[a-zA-Z_$/][\w$/]*(?:\$[\w$]+)?;", Name.Class),
            (r"[a-zA-Z_$/][\w$/]*(?:\.[<>$\w]+)?", Name),  # fallback for dotted names, <init>, etc.

            # Descriptors and method descriptors: ()V, (Ljava/lang/String;)I, [I, [[Ljava/lang/String;
            (r"\((?:\[*(?:[BCDFIJSZV]|L[^;]+;))*\)(?:\[*(?:[BCDFIJSZV]|L[^;]+;))", String.Symbol),
            (r"\[*(?:[BCDFIJSZV]|L[^;]+;)", String.Symbol),

            # String literals
            (r'"(\\\\|\\"|[^"])*"', String.Double),

            # Integers, hex, signed offsets
            (r"[-+]?(?:0x[0-9a-fA-F_]+|\d[\d_]*)", Number),

            # Punctuation & operators commonly present in annotated operands
            (r"[,:()\[\]{}.=]", Punctuation),
            (r"[-+*/%<>!&|^~?]", Operator),

            # Whitespace
            (r"\s+", Text),
        ],
    }
