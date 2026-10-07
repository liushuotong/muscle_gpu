#include "muscle.h"
#include "myutils.h"
#ifdef MUSCLE_GPU_FRONTEND
#include "gpu/backend_registry.h"
#endif
// Modified 2026-10-08: optional muscle_gpu frontend; original CLI parser retained.

string g_Arg1;

static void FrontendNotice(FILE *f)
	{
#ifdef MUSCLE_GPU_FRONTEND
	fputs("muscle_gpu: modified MUSCLE distribution (2026-10-08), GPLv3; see LICENSE.\n"
	      "Use the MUSCLE parameters below with executable muscle_gpu.\n"
	      "Default for -align/-super5/-super4: CUDA when available, otherwise reported CPU fallback.\n", f);
#else
	(void) f;
#endif
	}

int main(int argc, char **argv)
	{
	for (int i = 1; i < argc; ++i)
		{
		string s = string(argv[i]);
		if (s == "-h")
			{
			void Usage(FILE *f);
			Usage(stdout);
			FrontendNotice(stdout);
			return 0;
			}

		if (s == "-help" || s == "--help")
			{
			void Usage(FILE *f);
			Usage(stdout);
			FrontendNotice(stdout);
			return 0;
			}
		}

	MyCmdLine(argc, argv);
#ifdef MUSCLE_GPU_FRONTEND
	// Only the accelerated frontend changes the default backend. Explicit user
	// choices and the CPU batch regression switch always take precedence.
	if ((optset_align || optset_super5 || optset_super4) && !optset_backend && !optd(cpu_batch, false))
		{
		std::string reason;
		opt_backend = muscle_gpu::CudaAvailable(reason) ? "gpu" : "cpu";
		optset_backend = true;
		if (opt_backend == "cpu")
			ProgressLog("muscle_gpu: CPU fallback (%s)\n", reason.c_str());
		}
#endif
	if (!opt(quiet))
		{
		PrintBanner(stderr);
		FrontendNotice(stderr);
		if (argc < 2)
			return 0;
		}

	SetLogFileName(opt(log));
	LogProgramInfoAndCmdLine();

	extern vector<string> g_Argv;
	uint n = SIZE(g_Argv);
	asserta(n > 0);
	string ShortCmdLine;
	if (n > 1)
		ShortCmdLine = g_Argv[1];
	if (n > 2)
		{
		g_Arg1 = g_Argv[2];
		ShortCmdLine += " " + g_Argv[2];
		}
	if (n > 1)
		{
		ProgressPrefix(false);
		Progress("[%s]\n", ShortCmdLine.c_str() + 1);
		ProgressPrefix(true);
		}

	uint CmdCount = 0;
#define C(x)	if (optset_##x) ++CmdCount;
#include "cmds.h"
	if (CmdCount > 1)
		Die("More than one command specified");

#define C(x)	\
	if (optset_##x) \
		{ \
		g_Arg1 = opt_##x; \
		optused_##x = true; \
		void cmd_##x(); \
		cmd_##x(); \
		CheckUsedOpts(false); \
		LogElapsedTimeAndRAM(); \
		return 0; \
		}
#include "cmds.h"
#undef C

	return 0;
	}
