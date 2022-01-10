use std::error::Error;
use std::path::Path;
use serde::{Deserialize,Serialize};
use std::collections::HashMap;
use std::convert::From;
use clap::Parser;

/// a record for PacBio ipdSummary with in-silico model
#[derive(Debug, Deserialize)]
#[allow(non_snake_case)]
struct IpdSummary {
    refName: String,
    tpl: i64,
    strand: u8,
    base: char,
    score: u32,
    tMean: f32,
    tErr: f32,
    modelPrediction: f32,
    ipdRatio: f32,
    coverage: u32,
    frac: Option<f32>,
    fracLow: Option<f32>,
    fracUp: Option<f32>,
}

impl IpdSummary {
    fn into_pair(self) -> (IpdSummaryKey, IpdSummaryValue) {
        (IpdSummaryKey {
            refName: self.refName,
            tpl: self.tpl,
            strand: self.strand,
        }, IpdSummaryValue {
            base: self.base,
            score: self.score,
            tMean: self.tMean,
            tErr: self.tErr,
            modelPrediction: self.modelPrediction,
            ipdRatio: self.ipdRatio,
            coverage: self.coverage,
            frac: self.frac,
            fracLow: self.fracLow,
            fracUp: self.fracUp,
        })
    }
}

#[derive(Hash, Eq, PartialEq, Debug)]
#[allow(non_snake_case)]
struct IpdSummaryKey {
    refName: String,
    tpl: i64,
    strand: u8,
}

impl IpdSummaryKey {
    #[allow(non_snake_case)]
    fn new(refName: String, tpl: i64, strand: u8) -> Self {
        Self { refName, tpl, strand, }
    }

    /// return a new instance with an opposite strand
    fn opposite(&self) -> Self {
        Self {
            refName: self.refName.clone(),
            tpl: self.tpl,
            strand: match self.strand {
                0 => 1,
                1 => 0,
                n => panic!("Unexpected strand number: {}", n),
            }
        }
    }

    /// Extend IpdSummaryKey respecting its strand
    /// For a negative strand key, extension length `up` and `down` are swapped
    /// and keys in the reversed order are returned
    fn extend(&self, up: i64, down: i64) -> Box<dyn Iterator<Item = Self> + '_> {
        // ipdSummary: 1-based
        let position_left: i64;
        let position_right: i64;
        match self.strand {
            0 => {
                position_left = self.tpl.checked_sub(up)
                    .unwrap_or_else(||panic!("[ERROR] Target position overflowed. IpdSummary tpl: {}, extension length: {}", self.tpl, up));
                position_right = self.tpl.checked_add(down)
                    .unwrap_or_else(||panic!("[ERROR] Target position overflowed. IpdSummary tpl: {}, extension length: {}", self.tpl, down));
            },
            1 => {
                position_left = self.tpl.checked_sub(down)
                    .unwrap_or_else(||panic!("[ERROR] Target position overflowed. IpdSummary tpl: {}, extension length: {}", self.tpl, down));
                position_right = self.tpl.checked_add(up)
                    .unwrap_or_else(||panic!("[ERROR] Target position overflowed. IpdSummary tpl: {}, extension length: {}", self.tpl, up));
            },
            n => panic!("Unexpected strand: {}", n),
        };
        let range = position_left..=position_right;
        let keys = range.flat_map(|p| {
            [Self::new(self.refName.clone(), p, 0), Self::new(self.refName.clone(), p, 1)]
        });
        if self.strand == 0 { Box::new(keys) } else { Box::new(keys.rev()) }
    }
}

impl From<MergedOcc> for IpdSummaryKey {
    fn from(merged_occ: MergedOcc) -> Self {
        Self {
            refName: merged_occ.refName,
            // MergedOcc: 0-based, IpdSummary: 1-based
            tpl: merged_occ.start + 1,
            strand: match merged_occ.strand {
                '+' => 0,
                '-' => 1,
                c => panic!("Unexpected strand char: {}", c),
            },
        }
    }
}

#[derive(Debug, Default)]
#[allow(non_snake_case)]
#[allow(dead_code)]
struct IpdSummaryValue {
    base: char,
    score: u32,
    tMean: f32,
    tErr: f32,
    modelPrediction: f32,
    ipdRatio: f32,
    coverage: u32,
    frac: Option<f32>,
    fracLow: Option<f32>,
    fracUp: Option<f32>,
}

/// a record for a .merged_occ file, or a position list of motif occurrences
#[derive(Debug, Deserialize)]
#[allow(non_snake_case)]
struct MergedOcc {
    refName: String,
    /// 0-based position
    start: i64,
    strand: char,
}

#[derive(Debug, Clone, Serialize)]
struct TargetIpd {
    position: i64,
    strand: char,
    value: f32,
    label: String,
    src: i64,
}

impl TargetIpd {
    fn create_label(position: i64, region_width: i64, region_extension: i64, strand: char) -> String {
        let part = match position {
            p if p <= 0 => panic!("[ERROR] Position ({}) is smaller than 1", p),
            // start-side / upstream of the target region
            p if p <= region_extension => 's',
            // motif / target region
            p if p <= region_extension + region_width => 'm',
            // end-side / downstream of the target region
            p if p <= 2 * region_extension + region_width => 'e',
            p => panic!("[ERROR] Position ({}) is larger than the target region length", p),
        };
        let relative_position = match part {
            's' => position,
            'm' => position - region_extension,
            'e' => position - region_extension - region_width,
            _ => panic!("[ERROR] Unknown region part name"),
        };
        let label_strand = match strand {
            '+' => 'p',
            '-' => 'm',
            _ => panic!("[ERROR] Unknown strand"),
        };
        format!("{}{}{}", part, relative_position, label_strand)
    }

    fn new(position: i64, strand: char, value: f32, src: i64, region_width: i64, region_extension: i64) -> Self {
        Self {
            position,
            strand,
            value,
            label: Self::create_label(position, region_width, region_extension, strand),
            src,
        }
    }
}

fn collect_ipd_summary_in_merged_occ<P: AsRef<Path>>(
    kinetics_path: P, occ_path: P, occ_width: i64, occ_extension: i64, output_path: P) -> Result<(), Box<dyn Error>>
{
    let mut kinetics_reader = csv::Reader::from_path(kinetics_path)?;
    let mut occ_reader = csv::ReaderBuilder::new()
        .delimiter(b' ')
        .has_headers(false)
        .from_path(occ_path)?;
    let kinetics = kinetics_reader.deserialize::<IpdSummary>().map(|e| e.unwrap().into_pair()).collect::<HashMap<_,_>>();
    let default_ipd_summary_value = IpdSummaryValue::default();
    let target_kinetics = occ_reader.deserialize::<MergedOcc>().enumerate().flat_map(|(i, e)| {
        let target_key = IpdSummaryKey::from(e.unwrap());
        // generate key(-extension)..key(+width+extension) for each strand
        let target_keys = target_key.extend(occ_extension, occ_extension + occ_width - 1);
        let target_vals = target_keys.enumerate().map(|(j, key)| {
            let target_val = kinetics.get(&key).unwrap_or(&default_ipd_summary_value);
            let target_strand = if j % 2 == 0 { '+' } else { '-' };
            TargetIpd::new(((j / 2) + 1) as i64, target_strand, target_val.tMean, (i + 1) as i64, occ_width, occ_extension)
        }).collect::<Vec<_>>();
        if target_vals.len() as i64 != (occ_extension * 2 + occ_width) * 2 { panic!("Unexpected length of results for a motif occ"); }
        target_vals
    });
    let mut result_writer = csv::Writer::from_path(output_path)?;
    for target in target_kinetics {
        result_writer.serialize(target)?;
    }
    result_writer.flush()?;
    Ok(())
}

#[derive(Debug, Clone)]
struct RegionOverflow {
    message: String,
}
impl std::fmt::Display for RegionOverflow {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "RegionOverflow: {}", self.message)
    }
}
impl Error for RegionOverflow {}
impl std::default::Default for RegionOverflow {
    fn default() -> Self {
        RegionOverflow {
            message: "Total region length exceeds u64".to_string(),
        }
    }
}

/// Collect kinetics info at specified regions
#[derive(Debug, Parser)]
#[clap(about, version, author)]
struct Args {
    /// Kinetics CSV file generated by PacBio `ipdSummary`
    #[clap(long, short)]
    kinetics: String,

    /// File listing positions of motif occurrences or target bases.
    /// Each row has chromosome name, 0-based start position, and strand with delimiter of single
    /// space, without header line.
    #[clap(long)]
    occ: String,

    /// Length of the motif or target region including the start position
    #[clap(long)]
    occ_width: i64,

    /// Length of an extended region for each end of a target region
    #[clap(long)]
    extend: i64,

    /// Output CSV path
    #[clap(long, short)]
    output: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let kinetics_path = args.kinetics;
    let occ_path = args.occ;
    let occ_width = args.occ_width;
    let region_extension = args.extend;
    let output_path = args.output;
    // check if (region_extension * 2 + occ_width) overflows
    region_extension.checked_mul(2).ok_or(RegionOverflow::default())?.checked_add(occ_width).ok_or(RegionOverflow::default())?;
    collect_ipd_summary_in_merged_occ(kinetics_path, occ_path, occ_width, region_extension, output_path)?;
    Ok(())
}
