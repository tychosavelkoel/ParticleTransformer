#include <iostream>
#include <chrono>

#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TH1.h"

#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequenceArea.hh"

#include "include/ProgressBar.h"

#include "PU14/EventMixer.hh"
#include "PU14/CmdLine.hh"
#include "PU14/PU14.hh"

#include "include/extraInfo.hh"
#include "include/jetCollection.hh"
#include "include/softDropGroomer.hh"
#include "include/treeWriter.hh"
#include "include/jetMatcher.hh"
#include "include/Angularity.hh"

#include <cmath>

using namespace std;
using namespace fastjet;

// ./Data -hard samples/Train_JetMed-pthat_350-vac.res -nev 100000
// ./Data -hard samples/Test_JetMed-pthat_350-qhat.res -nev 100000


int main (int argc, char ** argv) {

  auto start_time = chrono::steady_clock::now();
  
  CmdLine cmdline(argc,argv);
  // inputs read from command line
  int nEvent = cmdline.value<int>("-nev",1);  // first argument: command line option; second argument: default value
  //bool verbose = cmdline.present("-verbose");

  cout << "will run on " << nEvent << " events" << endl;

  // Uncomment to silence fastjet banner
  ClusterSequence::set_fastjet_banner_stream(NULL);

  //to write info to root tree
  treeWriter trw("jetTree");
  TH1F *h1 = new TH1F("Width distribution", "", 50, 0, 1000);

  //Jet definition
  double R                   = 0.4;
  double ghostRapMax         = 6.0;
  double ghost_area          = 0.005;
  int    active_area_repeats = 1;
  GhostedAreaSpec ghost_spec(ghostRapMax, active_area_repeats, ghost_area);
  AreaDefinition area_def = AreaDefinition(active_area,ghost_spec);
  JetDefinition jet_def(antikt_algorithm, R);

  double jetRapMax = 3.0;
  Selector jet_selector = SelectorAbsRapMax(jetRapMax);

  Angularity width(1.,1.,R);
  Angularity pTD(0.,2.,R);
  Angularity mass(2.,1.,R);
    
  ProgressBar Bar(cout, nEvent);
  Bar.SetStyle(-1);

  EventMixer mixer(&cmdline);  //the mixing machinery from PU14 workshop

  // loop over events
  int iev = 0;
  unsigned int entryDiv = (nEvent > 200) ? nEvent / 200 : 1;
  while ( mixer.next_event() && iev < nEvent )
  {
    // increment event number
    iev++;

    Bar.Update(iev);
    Bar.PrintWithMod(entryDiv);

    vector<PseudoJet> particlesMergedAll = mixer.particles();

    vector<double> eventWeight;
    eventWeight.push_back(mixer.hard_weight());
    eventWeight.push_back(mixer.pu_weight());

    // extract hard partons that initiated the jets
    fastjet::Selector parton_selector = SelectorVertexNumber(-1);
    vector<PseudoJet> partons = parton_selector(particlesMergedAll);

    // select final state particles from hard event only
    fastjet::Selector sig_selector = SelectorVertexNumber(0);
    vector<PseudoJet> particlesSig = sig_selector(particlesMergedAll);

    // select final state particles from background event only
    fastjet::Selector bkg_selector = SelectorVertexNumber(1);
    vector<PseudoJet> particlesBkg = bkg_selector(particlesMergedAll);

    vector<PseudoJet> particlesMerged = particlesBkg;
    particlesMerged.insert( particlesMerged.end(), particlesSig.begin(), particlesSig.end() );
    
    
    //vector<PseudoJet> particlesBkg, particlesSig;
    //SelectorIsHard().sift(particlesMerged, particlesSig, particlesBkg); // this sifts the full event into two vectors of PseudoJet, one for the hard event, one for the underlying event

    //---------------------------------------------------------------------------
    //   jet clustering
    //---------------------------------------------------------------------------

    fastjet::ClusterSequenceArea csSig(particlesSig, jet_def, area_def);
    jetCollection jetCollectionSig(sorted_by_pt(jet_selector(csSig.inclusive_jets(390.)))); //select from which pt

        

    //cout << jetCollectionSig.getJet().size();

    for(PseudoJet jet : jetCollectionSig.getJet()) {
      double widthSig; //widthSig.reserve(jetCollectionSig.getJet().size());
      double pTDSig;   //pTDSig.reserve(jetCollectionSig.getJet().size());
      double massSig;   //massSig.reserve(jetCollectionSig.getJet().size());
      double nparticles;   //nparticles.reserve(jetCollectionSig.getJet().size());
      double label; //label.reserve(jetCollectionSig.getJet().size());
      double jet_energy; //jet_energy.reserve(jetCollectionSig.getJet().size());
      double jet_pt; //jet_pt.reserve(jetCollectionSig.getJet().size());
      double jet_eta; //jet_eta.reserve(jetCollectionSig.getJet().size());
      double jet_phi;
      
      vector<double> part_px; //part_px.reserve(jetCollectionSig.getJet().size());
      vector<double> part_py; //part_py.reserve(jetCollectionSig.getJet().size());
      vector<double> part_pz; //part_pz.reserve(jetCollectionSig.getJet().size());
      vector<double> part_pt; //part_pt.reserve(jetCollectionSig.getJet().size());
      vector<double> part_energy; //part_energy.reserve(jetCollectionSig.getJet().size());
      vector<double> part_deta; //part_deta.reserve(jetCollectionSig.getJet().size());
      vector<double> part_dphi; //part_dphi.reserve(jetCollectionSig.getJet().size());

      widthSig = width.result(jet);
      pTDSig = pTD.result(jet);
      massSig = mass.result(jet);
      label = 0; //use label 0 for unquenched jets
      jet_energy = jet.E();
      jet_pt= jet.perp();
      jet_eta = jet.eta();
      jet_phi = jet.phi_std(); //in range -pi to pi
      nparticles = jet.constituents().size();

      vector<double> px; px.reserve(jet.constituents().size());
      vector<double> py; py.reserve(jet.constituents().size());
      vector<double> pz; pz.reserve(jet.constituents().size());
      vector<double> pt; pt.reserve(jet.constituents().size());
      vector<double> energy; energy.reserve(jet.constituents().size());
      vector<double> deta; deta.reserve(jet.constituents().size());
      vector<double> dphi; dphi.reserve(jet.constituents().size());
      std::vector< fastjet::PseudoJet > constituents = jet.constituents();
      
      /* //only first 20 constituents + padding
      for (int i = 0; i < 20 ;i++){
        if (i < constituents.size()){
          px.push_back(constituents[i].px());
          py.push_back(constituents[i].py());
          pz.push_back(constituents[i].pz());

          pt.push_back(constituents[i].perp());
          E.push_back(constituents[i].E());
          deta.push_back(constituents[i].eta() - jet.eta());
          dphi.push_back(constituents[i].delta_phi_to(jet)); //in range -pi to pi
        }
        else {
          px.push_back(0);
          py.push_back(0);
          pz.push_back(0);

          pt.push_back(0);
          E.push_back(0);
          deta.push_back(0);
          dphi.push_back(0);
        }
      }
      */
      
      for(fastjet::PseudoJet& con : constituents) {
         
        //Option 1:
        //double pt = con.perp();
        //double phi = con.phi(); //phi in range 0,2pi, 
        //double eta = con.eta();
        //px.push_back(pt*cos(phi));
        //py.push_back(pt*sin(phi));
        //pz.push_back(pt*sinh(eta));      
        

        //Option 2:
        part_px.push_back(con.px());
        part_py.push_back(con.py());
        part_pz.push_back(con.pz());

        part_pt.push_back(con.perp());
        part_energy.push_back(con.E());
        part_deta.push_back(con.eta() - jet.eta());
        part_dphi.push_back(con.delta_phi_to(jet)); //in range -pi to pi
      }//end constituents loop

      /*
      part_px.push_back(px);
      part_py.push_back(py);
      part_pz.push_back(pz);
      part_pt.push_back(pt);
      part_energy.push_back(energy);
      part_deta.push_back(deta);
      part_dphi.push_back(dphi);
      */
      /*
      for(int i = 0; i < px.size(); i++){
        cout << px[i] << " ";
      }
      
      cout << px;
      */

      h1->Fill(jet_pt);
      trw.addSingle("width", widthSig);
      trw.addSingle("jet_pt", jet_pt);
      trw.addSingle("jet_mass", massSig);
      trw.addSingle("jet_energy", jet_energy);
      trw.addSingle("jet_eta", jet_eta);
      trw.addSingle("jet_phi", jet_phi);
      trw.addSingle("nparticles", nparticles);
      trw.addSingle("label", label);
      trw.addCollection("part_px", part_px);
      trw.addCollection("part_py", part_py);
      trw.addCollection("part_pz", part_pz);
      trw.addCollection("part_energy", part_energy);
      trw.addCollection("part_deta", part_deta);
      trw.addCollection("part_dphi", part_dphi);
      trw.fillTree();
    } //end jet loop
    
    /*
    jetCollectionSig.addVector("widthSig", widthSig);
    jetCollectionSig.addVector("pTDSig", pTDSig);
    jetCollectionSig.addVector("nparticles", nparticles);
    jetCollectionSig.addVector("label", label);
    jetCollectionSig.addVector("jet_energy", jet_energy);
    jetCollectionSig.addVector("jet_pt", jet_pt);
    
    jetCollectionSig.addVector("part_px", part_px);
    jetCollectionSig.addVector("part_py", part_py);
    jetCollectionSig.addVector("part_pz", part_pz);
    //jetCollectionSig.addVector("part_pt", part_pt); //defined in yaml file 
    jetCollectionSig.addVector("part_energy", part_energy);
    jetCollectionSig.addVector("part_deta", part_deta);
    jetCollectionSig.addVector("part_dphi", part_dphi);
    */
    //---------------------------------------------------------------------------
    //   Groom the jets
    //---------------------------------------------------------------------------

    //SoftDrop grooming classic for signal jets (zcut=0.1, beta=0)
    
    /*
    softDropGroomer sdgSigBeta00Z01(0.1, 0.0, R);
    jetCollection jetCollectionSigSDBeta00Z01(sdgSigBeta00Z01.doGrooming(jetCollectionSig));
    jetCollectionSigSDBeta00Z01.addVector("zgSigSDBeta00Z01",    sdgSigBeta00Z01.getZgs());
    jetCollectionSigSDBeta00Z01.addVector("ndropSigSDBeta00Z01", sdgSigBeta00Z01.getNDroppedSubjets());
    jetCollectionSigSDBeta00Z01.addVector("dr12SigSDBeta00Z01",  sdgSigBeta00Z01.getDR12());
    */

    //---------------------------------------------------------------------------
    //   write tree
    //---------------------------------------------------------------------------
    
    //Give variable we want to write out to treeWriter.
    //Only vectors of the types 'jetCollection', and 'double', 'int', 'PseudoJet' are supported

    //trw.addCollection("eventWeight",   eventWeight);
    //trw.addPartonCollection("partons",       partons);

    //trw.addCollection("sigJet",        jetCollectionSig);
    //trw.addCollection("sigJetSDBeta00Z01",      jetCollectionSigSDBeta00Z01);
    
   

  }//event loop

  Bar.Update(nEvent);
  Bar.Print();
  Bar.PrintLine();

  TTree *trOut = trw.getTree();

  TFile *fout = new TFile(cmdline.value<string>("-output", "Data_Train_390pt.root").c_str(), "RECREATE");
  trOut->Write();
  fout->Write();
  fout->SetCompressionAlgorithm(ROOT::kLZ4);
  fout->SetCompressionLevel(4);
  fout->Close();

  double time_in_seconds = chrono::duration_cast<chrono::milliseconds>
    (chrono::steady_clock::now() - start_time).count() / 1000.0;
  cout << "runFromFile: " << time_in_seconds << endl;
}
